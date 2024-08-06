import uvicorn
import os

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5720))
    uvicorn.run("mission:app", host="0.0.0.0", port=int(PORT), reload=True, workers=1)
    # uvicorn.run("mission:app", host="localhost", port=int(PORT), reload=True, debug=True, workers=1)
