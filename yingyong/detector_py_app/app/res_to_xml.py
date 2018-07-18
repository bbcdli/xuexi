#res_to_xml.py
import xml.etree.ElementTree as xml
import datetime
now = datetime.datetime.now()

min = now.minute
sec = now.second

filename = "result.xml"
root = xml.Element("Train")
root.set("name","Ideenzug")

kiwa = xml.SubElement(root,"MultifunctionArea")

num_element = xml.Element("Number")
num_element.text = "1"
kiwa.append(num_element)
OccupationState = xml.Element("OccupationState")

print('sec:',sec)
if (sec > 11 and sec < 20) or (sec > 41 and sec < 50):
    OccupationState.text = "FREE"
    OccupationType = xml.Element("OccupationType")
    OccupationType.text = "KINDERWAGEN"
    kiwa.append(OccupationState)
    kiwa.append(OccupationType)
else:
    OccupationState.text = "OCCUPIED"
    OccupationType = xml.Element("OccupationType")
    OccupationType.text = "KINDERWAGEN"
    kiwa.append(OccupationState)
    kiwa.append(OccupationType)

xml.dump(root)
tree = xml.ElementTree(root)
with open(filename, "w") as fh:
    tree.write(fh)

