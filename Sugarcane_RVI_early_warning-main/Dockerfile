FROM mambaorg/micromamba:1.5.8
COPY --chown=$MAMBA_USER:$MAMBA_USER dev/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes
WORKDIR /app
COPY . /app
EXPOSE 8501
CMD ["streamlit","run","dashboard.py","--server.address=0.0.0.0"]
