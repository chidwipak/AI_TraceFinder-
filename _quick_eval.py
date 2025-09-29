import os, random, cv2, numpy as np
from scanner_identification_fixed import FixedScannerIdentifier
from combined_feature_extractors import CombinedFeatureExtractor
import joblib

root = 'The SUPATLANTIQUE dataset'
flatfield = os.path.join(root, 'Flatfield')
official = os.path.join(root, 'Official')
wikipedia = os.path.join(root, 'Wikipedia')
tampered_root = os.path.join(root, 'Tampered images')

# Scanner evaluation
identifier = FixedScannerIdentifier()
if not identifier.load_model():
    print('Scanner model load failed'); raise SystemExit(1)

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
scanner_samples = scanner_samples[:50]

correct=0; total=0
for path, expected in scanner_samples:
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
    except Exception as e:
        pass

scanner_acc = (correct/total) if total else 0
print('Scanner: correct/total/acc = {}/{}/{:.3f}'.format(correct, total, scanner_acc))

# Tampered evaluation
feature_extractor = CombinedFeatureExtractor()
model = joblib.load('new_objective2_ensemble_results/models/best_ensemble_model.pkl')
scaler = joblib.load('new_objective2_ensemble_results/models/feature_scaler.pkl')

tampered_dir = os.path.join(tampered_root,'Tampered')
original_dir = os.path.join(tampered_root,'Original')

def gather_images(d):
    out=[]
    if os.path.isdir(d):
        for r,_,files in os.walk(d):
            for f in files:
                if f.lower().endswith(('.tif','.tiff','.png','.jpg','.jpeg')):
                    out.append(os.path.join(r,f))
    return out

T = gather_images(tampered_dir)
O = gather_images(original_dir)
random.shuffle(T); random.shuffle(O)
samples = [(p,1) for p in T[:50]] + [(p,0) for p in O[:50]]
random.shuffle(samples)

correct=0; total=0
for path,label in samples:
    try:
        if path.lower().endswith(('.tif','.tiff')):
            import tifffile
            img = tifffile.imread(path)
            if img.ndim==2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        feats = feature_extractor.extract_tampered_features(img).reshape(1,-1)
        feats = scaler.transform(feats)
        if hasattr(model,'predict_proba'):
            proba = model.predict_proba(feats)[0]
            pred = int(np.argmax(proba))
        else:
            pred = int(model.predict(feats)[0])
        total+=1
        correct += int(pred==label)
    except Exception:
        pass

tampered_acc = (correct/total) if total else 0
print('Tampered: correct/total/acc = {}/{}/{:.3f}'.format(correct, total, tampered_acc))
