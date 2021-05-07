FROM powerapi/powerapi:0.9.3

COPY --chown=powerapi . /tmp/selfwatts
RUN pip3 install --user --no-cache-dir "/tmp/selfwatts" && rm -r /tmp/selfwatts

ENTRYPOINT ["python3", "-m", "selfwatts"]
