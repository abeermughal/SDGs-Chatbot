# Use the official Python 3.11.9 image as a base
FROM python:3.11.9

ENV PYTHONUNBUFFERED=1


# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies from the requirements file
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Specify the main application file (app.py or main.py, depending on your setup)
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
