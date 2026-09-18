import os
import json
import secrets
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from app.core.config import settings

class APIKeyRecord(BaseModel):
    key_id: str
    api_key: str
    name: str
    created_at: str
    is_active: bool = True
    request_count: int = 0
    last_used_at: Optional[str] = None

    def to_public_dict(self) -> Dict[str, Any]:
        """Returns safe representation with partially masked key."""
        masked = self.api_key[:12] + "..." + self.api_key[-4:] if len(self.api_key) > 16 else self.api_key
        return {
            "key_id": self.key_id,
            "masked_key": masked,
            "full_key": self.api_key,
            "name": self.name,
            "created_at": self.created_at,
            "is_active": self.is_active,
            "request_count": self.request_count,
            "last_used_at": self.last_used_at
        }

class APIKeyService:
    """
    Manages Custom Model API Keys for the RAG Document Engine.
    Allows creating project-specific keys (e.g., Flutter, Web, Python backend),
    validating requests, and tracking usage.
    """

    def __init__(self, storage_file: Optional[Path] = None):
        self.storage_file = storage_file or settings.API_KEYS_FILE
        self._keys: Dict[str, APIKeyRecord] = {}
        self._load_keys()

    def _load_keys(self):
        if self.storage_file.exists():
            try:
                with open(self.storage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("keys", []):
                        rec = APIKeyRecord(**item)
                        self._keys[rec.api_key] = rec
            except Exception as e:
                print(f"[APIKeyService] Warning: Failed to read {self.storage_file}: {e}")

        # Ensure default master key exists
        master_key = settings.DEFAULT_MASTER_KEY
        if master_key not in self._keys:
            master_rec = APIKeyRecord(
                key_id="master-default",
                api_key=master_key,
                name="Master Project Key (Default)",
                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                is_active=True,
                request_count=0
            )
            self._keys[master_key] = master_rec
            self._save_keys()

    def _save_keys(self):
        try:
            self.storage_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "keys": [rec.model_dump() for rec in self._keys.values()]
            }
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[APIKeyService] Warning: Failed to write {self.storage_file}: {e}")

    def validate_key(self, api_key: str) -> Optional[APIKeyRecord]:
        """Validates an API key and updates its usage stats."""
        if not api_key:
            return None

        clean_key = api_key.strip()
        rec = self._keys.get(clean_key)
        if rec and rec.is_active:
            rec.request_count += 1
            rec.last_used_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._save_keys()
            return rec
        return None

    def create_key(self, name: str = "New Project Key") -> APIKeyRecord:
        """Generates a new custom model API key."""
        random_hex = secrets.token_hex(16)
        new_key = f"sk-rag-live-{random_hex}"
        key_id = f"key_{secrets.token_hex(4)}"

        record = APIKeyRecord(
            key_id=key_id,
            api_key=new_key,
            name=name,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            is_active=True,
            request_count=0
        )
        self._keys[new_key] = record
        self._save_keys()
        return record

    def list_keys(self) -> List[Dict[str, Any]]:
        """Lists all keys in public format."""
        return [rec.to_public_dict() for rec in self._keys.values()]

    def revoke_key(self, key_id: str) -> bool:
        """Deactivates an API key by key_id."""
        for rec in self._keys.values():
            if rec.key_id == key_id:
                rec.is_active = False
                self._save_keys()
                return True
        return False

    def get_primary_key(self) -> str:
        """Returns an active API key string for quick copy/pasting."""
        for rec in self._keys.values():
            if rec.is_active:
                return rec.api_key
        return settings.DEFAULT_MASTER_KEY

api_key_service = APIKeyService()
