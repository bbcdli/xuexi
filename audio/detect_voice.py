from tools_audio import VoiceActivityDetector
import argparse
import json
import os,sys


def save_to_file(data, filename):
  with open(filename, 'w') as fp:
    json.dump(data, fp)


if __name__ == "__main__":
  '''
  parser = argparse.ArgumentParser(
    description='Analyze input wave-file and save detected speech interval to json file.')
  parser.add_argument('inputfile', metavar='INPUTWAVE',
                      help='the full path to input wave file')
  parser.add_argument('outputfile', metavar='OUTPUTFILE',
                      help='the full path to output json file to save detected speech intervals')
  '''
#args = parser.parse_args()

inputfile = 'demo_j_mix7.wav'
outputfile = 'demo_j_sing.json'
v = VoiceActivityDetector(inputfile)
raw_detection = v.detect_speech()
speech_labels = v.convert_windows_to_readible_labels(raw_detection)

if not os.path.exists(outputfile):
  os.makedirs(outputfile)

print(speech_labels)
#save_to_file(speech_labels, args.outputfile)