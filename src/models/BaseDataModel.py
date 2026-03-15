from helpers.config import get_settings, Settings
from bson.objectid import ObjectId

class BaseDataModel:

    def __init__(self, db_client: object):
        self.db_client = db_client
        self.app_settings = get_settings()

    @staticmethod
    def to_object_id(id_str: str):
        """
        Convert string to ObjectId if valid, otherwise return as-is.
        Useful for supporting both system-generated and custom business IDs.
        """
        if not id_str:
            return id_str
        
        if isinstance(id_str, ObjectId):
            return id_str

        if ObjectId.is_valid(id_str):
            return ObjectId(id_str)
        
        return id_str