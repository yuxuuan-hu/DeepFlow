import json
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def process_case_name(name):
    """
    Process case name to extract meaningful words.
    """
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    name = re.sub(r'\d+D*', '', name)
    name = re.sub(r'\d+', '', name)
    name = re.sub(r'[^\w\s]', '', name)
    
    words = name.split()
    filtered_words = [word for word in words if len(word) > 1]
    filtered_words = [word for word in filtered_words if not re.match(r'^[a-zA-Z]$', word)]
    
    return ' '.join(filtered_words)

def generate_word_cloud(jsonl_file_path, output_image_path):
    """
    Read JSONL file and generate word cloud with custom background.
    """
    all_case_names = []
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'case_name' in data:
                        processed_name = process_case_name(data['case_name'])
                        all_case_names.append(processed_name)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse JSON in this line: {line.strip()}")

        if not all_case_names:
            print("Error: No 'case_name' found in file.")
            return

        text = ' '.join(all_case_names)

        wordcloud = WordCloud(
            width=1600,
            height=900,
            mode='RGBA',
            background_color=None,
            colormap='copper',
            max_words=200,
            collocations=False
        ).generate(text)

        plt.figure(figsize=(12, 6.75), facecolor=None)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)

        plt.savefig(output_image_path, dpi=300, transparent=True)

        print(f"Word cloud successfully saved to: {output_image_path}")
        plt.show()

    except FileNotFoundError:
        print(f"Error: File '{jsonl_file_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    jsonl_file = 'database/deepflow_agent.jsonl' 
    output_image = 'database/wordcloud.png'

    generate_word_cloud(jsonl_file, output_image)
