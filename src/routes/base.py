import logging
from routes.data import logger
from fastapi import FastAPI, APIRouter, Depends
import os
from helpers.config import get_settings, Settings
from time import sleep
import logging
logger=logging.getLogger("uvicorn.error")
base_router = APIRouter(
    prefix="/api/v1",
    tags=["api_v1"],
)

@base_router.get("/")
async def welcome(app_settings: Settings = Depends(get_settings)):

    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION

    return {
        "app_name": app_name,
        "app_version": app_version,
    }

@base_router.get("/send_reports")
async def send_reports(app_settings:Settings=Depends(get_settings)):

    # start the background task to send email reports
    task = send_email_reports.delay(mail_wait_seconds=5)

    for ix in range(5):
        logger.info(f"Sending report {ix+1}/5")
        sleep(3)
    return {"message": "Reports sent successfully", "task_id": task.id}