import numpy as np

np_filename = "/home/xiachen.wyh/work/data/face/data/features/part0_train.bin"
txt_filename = "gcn_test_data.txt"
data = np.fromfile(np_filename, dtype=np.float32).reshape(-1, 256)
# data = np.random.uniform(0.0,1.0,(500,256))
#data = np.random.randint(0,1,(500,25))
# sample_id = np.arange(data.shape[0])

print("Number of feature vectors: ", data.shape[0])
print("Dimention of a feature vector: ", data.shape[1])


# file = open(txt)
# np.savetxt(txt_filename,data,fmt='%d',delimiter=',',newline='\r\n')

print("Converting and saving data into "+txt_filename+"...")
file = open(txt_filename,"w")
N = data.shape[0]
for i in range(N):
    row = data[i,:]
    row_string = ""
    for j in range(row.shape[0]):
        if j > 0:
            row_string = row_string+","
        row_string = row_string+str(row[j])
    file.write(str(i+1)+"\t"+row_string+"\n")
file.close()
