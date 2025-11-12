# Use Node.js LTS on Debian Trixie (slim variant for smaller image size)
FROM tensorflow/tensorflow:2.20.0-jupyter


# Create and activate virtual environment outside workspace
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Add activation to shell profiles for interactive use
RUN echo 'source /opt/venv/bin/activate' >> /root/.bashrc
RUN echo 'source /opt/venv/bin/activate' >> /root/.profile

# Set the working directory
WORKDIR /workspace


# Keep container running for devcontainer usage
CMD ["sleep", "infinity"]