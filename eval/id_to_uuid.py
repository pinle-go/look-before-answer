import numpy as np 
import json
answers = json.load(open("answers.json"))
id_uuid = np.load("id_uuid.txt.npz")["id_uuid"].item()
new_answer= {} 
for key in answers: 
    new_answer[id_uuid[int(key)]] = answers[key]   

f = open("new_answer.json", "w") 
f.write(json.dumps(new_answer) )                                                                                                                    
