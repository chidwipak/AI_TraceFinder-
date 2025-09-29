import os, random, cv2
from scanner_identification_fixed import FixedScannerIdentifier

root = 'The SUPATLANTIQUE dataset'
flatfield = os.path.join(root, 'Flatfield')
official = os.path.join(root, 'Official')
wikipedia = os.path.join(root, 'Wikipedia')

identifier = FixedScannerIdentifier()
if not identifier.load_model():
    print('Scanner model load failed')
    raise SystemExit(1)

scanner_samples = []
for source in [flatfield, official, wikipedia]:
    if os.path.isdir(source):
        for scanner_dir in os.listdir(source):
            sp = os.path.join(source, scanner_dir)
            if not os.path.isdir(sp):
                continue
            dpi_candidates = []
            for dpi_dir in os.listdir(sp):
                if dpi_dir not in ('150','300'): continue
                dp = os.path.join(sp, dpi_dir)
                if not os.path.isdir(dp): continue
                imgs = [os.path.join(dp,f) for f in os.listdir(dp) if f.lower().endswith(('.tif','.tiff','.png','.jpg','.jpeg'))]
                if imgs:
                    dpi_candidates.append((dpi_dir, imgs))
            if dpi_candidates:
                dpi_dir, imgs = random.choice(dpi_candidates)
                random.shuffle(imgs)
                scanner_samples.append((imgs[0], scanner_dir))

random.shuffle(scanner_samples)
scanner_samples = scanner_samples[:20]

correct=0; total=0
for idx, (path, expected) in enumerate(scanner_samples, 1):
    try:
        if path.lower().endswith(('.tif','.tiff')):
            import tifffile
            img = tifffile.imread(path)
            if img.ndim==2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        pred, conf, status = identifier.predict_scanner(img)
        total+=1
        correct += int(pred==expected)
        print(f"[{idx:02d}/20] expected={expected} predicted={pred} conf={conf:.3f}")
    except Exception as e:
        print(f"[{idx:02d}/20] error: {e}")

acc = (correct/total) if total else 0
print('Scanner: correct/total/acc = {}/{}/{:.3f}'.format(correct, total, acc))
