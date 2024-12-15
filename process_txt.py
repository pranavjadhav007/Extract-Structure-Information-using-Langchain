import os

directory_path = "./story/"

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    
    if filename.endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            
            updated_content = content.replace("”", '"').replace("“", '"')
            
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(updated_content)
            
            print(f"Updated file: {filename}")
        
        except Exception as e:
            print(f"Error for {filename} as {e}")
