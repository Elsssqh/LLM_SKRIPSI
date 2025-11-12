import json
from bs4 import BeautifulSoup
import re # Diperlukan untuk membersihkan teks

def create_error_dataset_from_sgml(sgml_file_path, output_json_path):
    """
    Mem-parsing file SGML dari dataset TOCFL dan mengubahnya menjadi
    dataset training untuk deteksi kesalahan (Error Localization Module).
    
    Args:
        sgml_file_path (str): Path ke file .sgml (misal: 'TOCFL-Grammar-A2.sgml')
        output_json_path (str): Nama file JSON untuk menyimpan hasil (misal: 'dataset_error.json')
    """
    
    print(f"Mulai mem-parsing file: {sgml_file_path}...")
    
    try:
        # Buka dan baca file SGML
        with open(sgml_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"ERROR: File tidak ditemukan di {sgml_file_path}")
        return
    except Exception as e:
        print(f"ERROR saat membaca file: {e}")
        return

    # Inisialisasi BeautifulSoup
    # Kita gunakan 'html.parser' karena lebih toleran terhadap tag SGML
    soup = BeautifulSoup(content, 'html.parser')
    
    training_data = []
    
    # 1. Temukan semua tag <ESSAY>
    essays = soup.find_all('essay')
    
    if not essays:
        print("ERROR: Tidak ada tag <ESSAY> yang ditemukan. Periksa format file Anda.")
        return

    print(f"Menemukan {len(essays)} esai...")

    # 2. Loop setiap esai
    for essay in essays:
        
        # 3. Ekstrak Teks Input (X)
        text_node = essay.find('text')
        if not text_node:
            continue
            
        # Ambil teks mentah dan bersihkan dari tag <p>
        # Kita gabungkan semua paragraf menjadi satu string
        paragraphs = text_node.find_all('p')
        if paragraphs:
            input_text = "\n".join([p.get_text(strip=True) for p in paragraphs])
        else:
            # Fallback jika tidak ada tag <p>
            input_text = text_node.get_text(strip=True)
            
        # Bersihkan spasi ganda atau karakter aneh jika ada
        input_text = re.sub(r'\s+', ' ', input_text).strip()
        
        if not input_text:
            continue # Lewati esai kosong

        # 4. Ekstrak Target Output (Y) - Daftar Kesalahan
        error_list = []
        
        for mistake in essay.find_all('mistake'):
            error_type_node = mistake.find('type')
            correction_node = mistake.find('correction')
            
            # Ambil data dari tag
            err_type = error_type_node.get_text(strip=True) if error_type_node else "N/A"
            correction = correction_node.get_text(strip=True) if correction_node else "N/A"
            
            # Ambil data dari atribut tag <MISTAKE>
            start_pos = mistake.get('start_off', 'N/A')
            end_pos = mistake.get('end_off', 'N/A')
            
            # Ekstrak fragmen yang salah dari teks input menggunakan posisi
            incorrect_fragment = "N/A"
            try:
                # Konversi posisi ke integer
                start = int(start_pos)
                end = int(end_pos)
                incorrect_fragment = input_text[start:end]
            except:
                pass # Biarkan 'N/A' jika ada masalah konversi

            error_entry = {
                "error_type": err_type,
                "error_position": f"({start_pos}, {end_pos})",
                "incorrect_fragment": incorrect_fragment,
                "suggested_correction": correction
            }
            error_list.append(error_entry)
            
        # 5. Buat string JSON untuk target 'Y'
        output_json_string = json.dumps(error_list, ensure_ascii=False)
        
        # 6. Tambahkan ke dataset training
        training_data.append({
            "input": input_text,
            "output": output_json_string
        })

    # 7. Simpan dataset ke file JSON
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        print(f"\nBerhasil! {len(training_data)} data training telah disimpan ke '{output_json_path}'")
    except Exception as e:
        print(f"\nERROR: Gagal menyimpan file JSON: {e}")

# --- CARA MENJALANKAN SKRIP ---

# 1. Ganti nama file ini dengan path ke data TOCFL Anda
file_a2 = 'DATASET\TOCFL-Grammar-A2.sgml' # (Ini untuk HSK 1-2)
file_b1 = 'DATASET\TOCFL-Grammar-A2.sgml' # (Ini untuk HSK 3)

# 2. Jalankan parser untuk kedua file
create_error_dataset_from_sgml(file_a2, 'dataset_error_A2.json')
create_error_dataset_from_sgml(file_b1, 'dataset_error_B1.json')

# 3. (Opsional) Anda bisa menggabungkan kedua file JSON tersebut menjadi satu
#    dataset training yang besar untuk 'error_prompt.pt' Anda.