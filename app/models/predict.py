import numpy as np
from pickle import load

tf_prt = load(open('./app/models/tf_prt.pkl', 'rb'))
tf_lig = load(open('./app/models/tf_lig.pkl', 'rb'))
model  = load(open('./app/models/model.pkl', 'rb'))

def predict(sequences, smiles):
    seq_ftr = tf_prt.transform(sequences).todense()
    sml_ftr = tf_lig.transform(smiles).todense()
    Z = np.zeros((len(sequences) * len(smiles), 200))
    for i in range(len(sequences)):
        for j in range(len(smiles)):
            Z[len(smiles)*i+j,:100] = seq_ftr[i,:]
            Z[len(smiles)*i+j,100:] = sml_ftr[j,:]
    predictions = model.predict_proba(Z)[:,1]
    P = np.zeros((len(sequences), len(smiles)))
    for i in range(len(sequences)):
        for j in range(len(smiles)):
            P[i,j] = predictions[len(smiles)*i+j]
    return {
        'proteins': sequences,
        'drugs': smiles,
        'predictions': P.tolist()
    }
