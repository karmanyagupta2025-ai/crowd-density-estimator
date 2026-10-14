from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder

# YOLO image prediction
from backend.app.services.predictor import run_prediction, MODEL_TYPE
# Video processing
from backend.app.services.video_predictor import process_video
# MySQL connection
from backend.app.database import get_connection
import base64
import io
import traceback
import os
# Create the outputs folder if it doesn't already exist
os.makedirs("outputs", exist_ok=True)
# Create the FastAPI application
app = FastAPI()
# ---------------------------------------------------------
# CORS CONFIGURATION (allow frontend to communicate with the backend (differing ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# STATIC FILES
# Makes files inside "outputs" accessible through /outputs/...
app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
    name="outputs"
)
#BACKEND CHECK
# HEALTH CHECK
# Used to check whether the FastAPI backend is running.
@app.get("/")
def health():
    return {"status": "CrowdSense backend running"}
# GET ACTIVE CAMERAS
# The frontend can call this endpoint to get the available
# cameras and their corresponding zones/capacity limits.
@app.get("/cameras")
def get_cameras():
    conn = None
    cursor = None
    try:
        # Connect to MySQL
        conn = get_connection()
        # dictionary=True makes database rows easier to use:
        # camera["camera_id"], camera["zone_name"], etc.
        cursor = conn.cursor(dictionary=True)
        # Get camera information together with its zone information
        cursor.execute("""
            SELECT
                c.camera_id,
                c.location_name,
                c.status,
                z.zone_id,
                z.zone_name,
                z.capacity_limit
            FROM camera c
            JOIN zone z
                ON c.zone_id = z.zone_id
            WHERE c.status = 'Active'
            ORDER BY c.camera_id
        """)
        return cursor.fetchall()
    except Exception as e:
        #  HTTP 500 error if something goes wrong
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load cameras: {str(e)}"
        )
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# DETECTION HISTORY
@app.get("/detection-history")
def get_detection_history():
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                d.event_id,
                d.timestamp,
                d.person_count,
                c.camera_id,
                c.location_name,
                z.zone_name,
                z.capacity_limit,
                CASE
                    WHEN d.person_count > z.capacity_limit
                    THEN 'Overcrowding'
                    ELSE 'Normal'
                END AS status
            FROM detection_event d
            JOIN camera c ON d.camera_id = c.camera_id
            JOIN zone z ON c.zone_id = z.zone_id
            ORDER BY d.timestamp DESC, d.event_id DESC
            LIMIT 100
        """)

        return jsonable_encoder(cursor.fetchall())

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load detection history: {str(e)}"
        )
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# ALERT HISTORY
@app.get("/alert-history")
def get_alert_history():
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                a.alert_id,
                a.triggered_at,
                a.alert_type,
                a.message,
                d.event_id,
                d.person_count,
                c.location_name,
                z.zone_name,
                z.capacity_limit
            FROM alert a
            JOIN detection_event d ON a.event_id = d.event_id
            JOIN camera c ON d.camera_id = c.camera_id
            JOIN zone z ON c.zone_id = z.zone_id
            ORDER BY a.triggered_at DESC, a.alert_id DESC
            LIMIT 100
        """)

        return jsonable_encoder(cursor.fetchall())

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load alert history: {str(e)}"
        )
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# IMAGE PREDICTION
# Receives:
#   1. An image
#   2. The camera_id selected by the user
#
# Then:
#   Image → YOLO → Person Count → MySQL
#                     ↓
#                  Trigger
#                     ↓
#                   Alert
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    camera_id: int = Form(...)
):
    # 1. VALIDATE IMAGE
    # Make sure the uploaded file is actually an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are accepted"
        )
    # Read the uploaded image into memory
    contents = await file.read()
    # Reject an empty upload
    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Empty file received"
        )
    conn = None
    cursor = None
    try:
        # 2. CONNECTION TO MYSQL
        conn = get_connection()
        # dict=True lets us access columns by name
        cursor = conn.cursor(dictionary=True)

        # 3. GET CAMERA + ZONE INFORMATION
        # use camera_id to find:
        # camera → location → zone → capacity
        # avoids storing repeated location/zone data
        # inside DETECTION_EVENT.
        cursor.execute("""
            SELECT
                c.camera_id,
                c.location_name,
                c.zone_id,
                z.zone_name,
                z.capacity_limit
            FROM camera c
            JOIN zone z
                ON c.zone_id = z.zone_id
            WHERE c.camera_id = %s
              AND c.status = 'Active'
        """, (camera_id,))
        camera = cursor.fetchone()
        # If the camera doesn't exist or is inactive,
        # stop processing.
        if not camera:
            raise HTTPException(
                status_code=404,
                detail="Invalid or inactive camera"
            )
        # 4.YOLO
        # This uses your EXISTING YOLO code.
        # We are not changing the model.
        # result_img = processed image/heatmap
        # count      = number of people detected
        result_img, count = run_prediction(contents)
        # 5. INSERT DETECTION INTO MYSQL
        # Store the YOLO result in DETECTION_EVENT.

        # camera_id connects this detection to:
        # CAMERA → ZONE
        cursor.execute("""
            INSERT INTO detection_event
            (timestamp, person_count, camera_id)
            VALUES (NOW(), %s, %s)
        """, (count, camera_id))
        # Get the automatically generated event_id
        event_id = cursor.lastrowid
        # 6. COMMIT THE INSERT
        # This permanently saves the detection event.
        #
        # IMPORTANT:
        #  MySQL trigger runs automatically when this
        # INSERT happens.
        # If person_count > capacity_limit:
        #       ALERT is automatically created.
        conn.commit()

        # 7. CHECK WHETHER AN ALERT WAS CREATED
        # The trigger may have inserted an alert.
        #  checking using the event_id.
        cursor.execute("""
            SELECT
                alert_id,
                alert_type,
                message,
                triggered_at
            FROM alert
            WHERE event_id = %s
        """, (event_id,))
        alert = cursor.fetchone()
        # 8. CONVERT TO BASE64

        # Base64 data.
        buffer = io.BytesIO()

        result_img.save(
            buffer,
            format="PNG"
        )
        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")
        # 9. SENDING RESULT BACK TO FRONTENd

        return JSONResponse(
            content=jsonable_encoder({
                # Database event ID
                "event_id": event_id,

                # YOLO person count
                "count": count,

                # Model being used
                "model": MODEL_TYPE,

                # Camera information
                "camera_id": camera["camera_id"],
                "location_name": camera["location_name"],

                # Zone information
                "zone_id": camera["zone_id"],
                "zone_name": camera["zone_name"],

                # Maximum allowed crowd
                "capacity_limit": camera["capacity_limit"],

                # Alert generated by MySQL trigger
                "alert": alert,

                # Processed image / heatmap
                "heatmap_base64": f"data:image/png;base64,{encoded}"
            })
        )


    # HANDLE EXPECTED HTTP ERRORS
    except HTTPException:
        # Undo any uncommitted database changes
        if conn:
            conn.rollback()
        raise
    # HANDLE OTHER ERRORS
    except Exception as e:
        # Undo database changes if something failed
        if conn:
            conn.rollback()
        # Print detailed error in terminal
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )
    # ALWAYS close DB connect
    finally:

        if cursor:
            cursor.close()

        if conn:
            conn.close()

# VIDEO PREDICTION
# keep existing video-processing functionality.
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):

    # Make sure the uploaded file is a video
    if not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="Only video files are accepted"
        )
    try:
        # Save uploaded video inside outputs/
        input_path = f"outputs/{file.filename}"
        with open(input_path, "wb") as f:
            f.write(await file.read())
        # Define the processed video output path
        output_path = f"outputs/processed_{file.filename}"
        # Run your existing video processing function
        process_video(
            input_path,
            output_path
        )
        # Return the location of the processed video
        return JSONResponse({
            "message": "Video processed successfully",
            "output_video": f"/outputs/processed_{file.filename}"
        })
    except Exception as e:
        # Print detailed error in terminal
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )