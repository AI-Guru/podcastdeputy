import uvicorn
import sys

if __name__ == "__main__":
    uvicorn.run("aiupdate_app:app", host="0.0.0.0", port=8202, reload=True)
