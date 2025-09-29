import os, random, cv2, numpy as np
from combined_feature_extractors import CombinedFeatureExtractor
import joblib

root = 'The SUPATLANTIQUE dataset'
tampered_root = os.path.join(root, 'Tampered images')

feature_extractor = CombinedFeatureExtractor()
model = joblib.load('new_objective2_ensemble_results/models/best_ensemble_model.pkl')
scaler = joblib.load('new_objective2_ensemble_results/models/feature_scaler.pkl')

def gather_images(d):
    out=[]
    if os.path.isdir(d):
        for r,_,files in os.walk(d):
            for f in files:
                if f.lower().endswith(('.tif','.tiff','.png','.jpg','.jpeg')):
                    out.append(os.path.join(r,f))
    return out

T = gather_images(os.path.join(tampered_root,'Tampered'))
O = gather_images(os.path.join(tampered_root,'Original'))
random.shuffle(T); random.shuffle(O)
samples = [(p,1) for p in T[:10]] + [(p,0) for p in O[:10]]
random.shuffle(samples)

correct=0; total=0
for idx,(path,label) in enumerate(samples,1):
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
            conf = float(np.max(proba))
        else:
            pred = int(model.predict(feats)[0])
            conf = 0.5
        total+=1
        correct += int(pred==label)
        print(f"[{idx:02d}/20] expected={'Tampered' if label==1 else 'Original'} predicted={'Tampered' if pred==1 else 'Original'} conf={conf:.3f}")
    except Exception as e:
        print(f"[{idx:02d}/20] error: {e}")

acc = (correct/total) if total else 0
print('Tampered: correct/total/acc = {}/{}/{:.3f}'.format(correct, total, acc))
