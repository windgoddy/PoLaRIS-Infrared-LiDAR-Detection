#!/usr/bin/env python3
"""
Parse cat3_difficulty_analysis.txt and create JSON file
"""
import json
import re

def parse_difficulty_txt(txt_file):
    """Parse the difficulty analysis text file"""
    easy_samples = []
    medium_samples = []
    hard_samples = []
    all_scores = []

    current_category = None

    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()

            # Detect category headers
            if line.startswith('Easy ('):
                current_category = 'easy'
                continue
            elif line.startswith('Medium ('):
                current_category = 'medium'
                continue
            elif line.startswith('Hard ('):
                current_category = 'hard'
                continue

            # Parse sample lines (format: "  img_id  IoU: 0.xxxx  Thresh: 0.x")
            if current_category and '  IoU:' in line:
                # Extract img_id, iou, and threshold using regex
                match = re.match(r'\s*(\S+)\s+IoU:\s+(\d+\.\d+)\s+Thresh:\s+(\d+\.\d+)', line)
                if match:
                    img_id = match.group(1)
                    iou = float(match.group(2))
                    thresh = float(match.group(3))

                    # Add to appropriate category
                    if current_category == 'easy':
                        easy_samples.append(img_id)
                    elif current_category == 'medium':
                        medium_samples.append(img_id)
                    elif current_category == 'hard':
                        hard_samples.append(img_id)

                    # Add to scores list
                    all_scores.append({
                        'img_id': img_id,
                        'box_iou': iou,
                        'best_thresh': thresh
                    })

    return {
        'easy': easy_samples,
        'medium': medium_samples,
        'hard': hard_samples,
        'scores': all_scores
    }

if __name__ == '__main__':
    print("Parsing cat3_difficulty_analysis.txt...")

    data = parse_difficulty_txt('cat3_difficulty_analysis.txt')

    print(f"  ✓ Easy: {len(data['easy'])} samples")
    print(f"  ✓ Medium: {len(data['medium'])} samples")
    print(f"  ✓ Hard: {len(data['hard'])} samples")
    print(f"  ✓ Total scores: {len(data['scores'])} samples")

    # Save JSON
    output_file = 'cat3_difficulty_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Created {output_file}")
