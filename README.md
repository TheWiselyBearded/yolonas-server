# YOLO-NAS Server

This repository contains a YOLO-NAS server instance that receives input frames, runs the image through the model, and streams a JSON file containing the results to a client via sockets.

It can be paired with a Unity project to receive virtual camera frames, run object detection on them, and transmit the results to a Unity client.

## Prerequisites

- Python 3.7 or higher
- Conda package manager (installation instructions: [https://docs.conda.io/projects/conda/en/latest/user-guide/install/](https://docs.conda.io/projects/conda/en/latest/user-guide/install/))

## Installation

1. Clone this repository:

   ```shell
   git clone https://github.com/your-username/yolo-server.git
   ```

2. Navigate to the project directory:

  ```shell
  cd yolo-server
  ```
  
3. Create a new Conda environment using the provided environment.yml file:

  ```shell
  conda env create -f environment.yml
  ```

4. Activate the Conda environment:

  ```shell
  conda activate yolo-nas
  ```
  
## Usage
Start the YOLO-NAS server:

  ```shell
  python yolo-server.py
  ```

The server will start listening for incoming frames and perform object detection on them.

Set up your Unity project to send frames to the server and receive the results. Example code and instructions can be found in the Unity project repository: https://github.com/TheWiselyBearded/ChatbotAvatarAI

## Contributing

Contributions are welcome! If you find a bug or have an idea for an improvement, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
