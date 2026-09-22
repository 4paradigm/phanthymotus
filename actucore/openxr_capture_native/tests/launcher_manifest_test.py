"""Keep the 2D enrollment launcher outside the immersive activity."""
from pathlib import Path
import xml.etree.ElementTree as E
base=Path(__file__).resolve().parents[1]/'app/src'
a='{http://schemas.android.com/apk/res/android}'
app=E.parse(base/'main/AndroidManifest.xml').getroot().find('application')
launcher=next(x for x in app.findall('activity') if x.get(a+'name').endswith('ConnectionActivity'))
assert {x.get(a+'name') for x in launcher.iter('category')}=={'android.intent.category.LAUNCHER'}
pico=E.parse(base/'pico/AndroidManifest.xml').getroot().find('application')
assert not pico.findall('meta-data')
native=pico.find('activity')
assert native.get(a+'name')=='android.app.NativeActivity'
assert any(x.get(a+'name')=='pvr.app.type' and x.get(a+'value')=='vr' for x in native.findall('meta-data'))
print('Launcher manifest checks PASS')
