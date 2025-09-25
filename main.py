#!/usr/bin/env python3
"""
BookBot Full - Entry Point
Чистая слоевая архитектура
"""

from app.interfaces.main_api import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 