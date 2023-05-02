import pandas as pd
import os
import chardet
# define a list of file paths
file_paths = ['file1.csv', 'file2.csv', 'file3.csv']
words = pd.read_csv("D:/bitirme_dataset/train/SignList_ClassId_TR_EN.csv",sep= ",",encoding = "latin5")
df = pd.read_csv("D:/bitirme_dataset/test/test_labels/ground_truth.csv",sep = ",",encoding = "latin5")

new_list = []

# loop through the numbers 0 to 225
for i in range(226):
    # check if the number exists in the '133' column
    if i in df['133'].values:
        # append the values from the first column of the rows that match the number to the new list
        new_list.append(list(df.loc[df['133'] == i].iloc[:, 0])[0])

# print the new list
print(new_list)
# Extract the values from the second column and store them in a list
col2_values = list(words.iloc[:, 1])

# Print the list of values
print(col2_values)






for i in range(226):
    with open("D:/bitirme_dataset/test/test/"+new_list[i]+"_color.mp4", 'rb') as source_file:
        # read the contents of the file as bytes
        file_contents = source_file.read()

    # open a new file for writing in binary mode
    with open("D:/bitirme_dataset/voice_to_video/"+col2_values[i]+".mp4", 'wb') as new_file:
        # write the contents of the source file to the new file
        new_file.write(file_contents)
