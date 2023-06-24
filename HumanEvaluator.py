import streamlit as st
import json
import base64



def get_binary_file_downloader(data, file_label='File'):
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}.json">Download JSON File</a>'
    return href
    
    

# Load the JSON file
with open('mydata.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert the keys to integers
data = {int(k): v for k, v in data.items()}
user_inputs={}
for i in range(0,399):
    user_inputs[str(i)]={
        "Id": str(i),
        "dataset" : "Xsum",
        "facebook/bart-large-cnn" : "8",
        "google/pegasus-xsum" : "8",
        "t5-large" : "8",
        "gpt-3.5-turbo" : "8"   
              
    }
#st.write(user_inputs)

# Get the current index from the button clicks
index = st.session_state.get('index', 0)
if st.button('Previous'):
    index = max(index - 1, 0)
    st.session_state.index = index
if st.button('Next'):
    index = min(index + 1, max(data.keys()))
    st.session_state.index = index

# Display the data for the current index
st.write(f'Entry {index}:')
entry = data[index]

# New dictionary to store user inputs
#user_inputs = st.session_state.get('user_inputs', {})
skiplist=['Id','dataset','InputText','ReferenceSummary']


# Initialize sub-dictionary for this index if it doesn't exist
if index not in user_inputs:
    user_inputs[index] = {}

for key, value in entry.items():
  if(key not in skiplist):

    st.write(f'{key}: {value}')
    # Create input boxes for user to enter scores
    user_input = st.number_input(f'Enter a number for {key}', min_value=0, max_value=10)
    user_inputs[index][key] = user_input

st.session_state.user_inputs = user_inputs

# Save button to save user inputs to another json file
if st.button('Save'):
    with open('user_inputs.json', 'w', encoding='utf-8') as f:
        json.dump(user_inputs, f, ensure_ascii=False, indent=4)
        
    with open('user_inputs.json', 'rb') as f:
        bytes_data = f.read()

    st.markdown(get_binary_file_downloader(bytes_data, 'user_inputs'), unsafe_allow_html=True)


