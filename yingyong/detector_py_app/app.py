#app.py
from flask import Flask, make_response, render_template
import analyzer
import argparse
import signal
import os


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", required=True)
args = parser.parse_args()

if not os.path.exists(args.config):
    raise IOError("config file does not exist!!!")

detector = analyzer.Analyzer(args.config)


def gracefully_exit(signal, frame):
    print("Stopping all threads and exiting...")
    detector.stop()
    exit()


app = Flask(__name__)

signal.signal(signal.SIGINT, gracefully_exit)
signal.signal(signal.SIGTERM, gracefully_exit)


@app.route("/cgi-bin/result.xml")
def return_result():
    state = detector.get_state()
    temp_xml = render_template('template.xml', val=state) #val is set in xml template as {{val}}
    response = make_response(temp_xml)
    response.headers["Content-Type"] = "application/xml"
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)

