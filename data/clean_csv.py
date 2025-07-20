import csv

with open("outlets.csv", "r", encoding="cp1252") as infile, \
     open("outlets_fixed.csv", "w", encoding="utf-8", newline="") as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        cleaned_row = [
            col.replace("�", "-")  # Replace with dash or remove
               .replace("?", "")   # Remove stray question marks
               .replace("–", "-")  # Normalize en dash to hyphen
               .replace("—", "-")  # Normalize em dash to hyphen
               .strip()
            for col in row
        ]
        writer.writerow(cleaned_row)

print("✅ CSV cleaned and saved as 'outlets_fixed.csv'")
