# How to run easily
Use Docker to have repeatable success.  If you can install Docker, you win.

### Install Docker
https://docs.docker.com/docker-for-mac/install/
https://docs.docker.com/docker-for-windows/install/

### Setup
cd foto2vam
docker build -t vdo/foto2vam:1.0.0 .

### Running for new models
cd foto2vam
Add new photos to Input folder then....
docker run -v $(pwd)/:/var/app/ vdo/foto2vam:1.0.0
