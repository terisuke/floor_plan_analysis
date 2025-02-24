import os
import shutil

def organize_floor_files():
    # dataディレクトリのパス
    data_dir = 'data'
    
    # 1Fと2Fのディレクトリを作成
    first_floor_dir = os.path.join(data_dir, '1F')
    second_floor_dir = os.path.join(data_dir, '2F')
    
    os.makedirs(first_floor_dir, exist_ok=True)
    os.makedirs(second_floor_dir, exist_ok=True)
    
    # dataディレクトリ内のファイルを走査
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            source_path = os.path.join(data_dir, filename)
            
            # ファイル名に基づいて振り分け
            if '1F.csv' in filename:
                dest_path = os.path.join(first_floor_dir, filename)
                shutil.move(source_path, dest_path)
            elif '2F.csv' in filename:
                dest_path = os.path.join(second_floor_dir, filename)
                shutil.move(source_path, dest_path)

if __name__ == '__main__':
    organize_floor_files() 