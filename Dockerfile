# Stage 1: Use the official Python slim image as a parent image
# Using a -slim image is a good practice as it contains the minimal packages
# needed to run Python, resulting in a smaller final image size.
FROM python:3.12.6-slim

# Set the working directory in the container to /app
# This is where your application's code will live inside the container.
# All subsequent commands (like COPY and CMD) will be run from this directory.
WORKDIR /

# Copy the requirements.txt file into the working directory
# We copy this file first to leverage Docker's layer caching.
# The pip install step will only be re-run if this file changes.
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# The --no-cache-dir option tells pip not to store the downloaded packages,
# which helps keep the image size down.
RUN pip install --timeout=300 --no-cache-dir -r requirements.txt

# Copy the rest of the application's code (e.g., main.py and any other modules)
# into the working directory. The '.' means copy from the host's current
# directory to the container's WORKDIR (/app).
COPY . .

EXPOSE 9000

# Command to run the application
# Use uvicorn to run the FastAPI app.
# --host 0.0.0.0 makes the server accessible from outside the container.
# --reload is removed as it's intended for development, not production.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]