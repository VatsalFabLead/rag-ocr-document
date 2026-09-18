from typing import List
from app.models.ocr_result import WordBox, LineBox, BoundingBox

class DocumentTokenizer:
    """
    Groups recognized WordBoxes into structured lines and paragraphs
    based on spatial geometry, line alignment, and word spacing.
    """

    def __init__(self, line_y_overlap_ratio: float = 0.5, max_word_gap_ratio: float = 2.5):
        self.line_y_overlap_ratio = line_y_overlap_ratio
        self.max_word_gap_ratio = max_word_gap_ratio

    def group_into_lines(self, words: List[WordBox], img_w: int = 1, img_h: int = 1) -> List[LineBox]:
        """
        Groups recognized words into ordered lines.
        """
        if not words:
            return []

        # Sort words primarily by vertical Y position, then X position
        sorted_words = sorted(words, key=lambda w: (w.bbox.y, w.bbox.x))

        lines: List[List[WordBox]] = []
        current_line: List[WordBox] = [sorted_words[0]]

        for word in sorted_words[1:]:
            # Word center-Y and line median center-Y
            word_cy = word.bbox.y + (word.bbox.height / 2.0)
            line_cy = sum(w.bbox.y + (w.bbox.height / 2.0) for w in current_line) / len(current_line)
            avg_height = sum(w.bbox.height for w in current_line) / len(current_line)

            # Check if word aligns vertically with current line
            if abs(word_cy - line_cy) <= (avg_height * 0.65):
                # Same line
                current_line.append(word)
            else:
                # Start new line
                lines.append(sorted(current_line, key=lambda w: w.bbox.x))
                current_line = [word]

        if current_line:
            lines.append(sorted(current_line, key=lambda w: w.bbox.x))

        # Build LineBox objects
        line_boxes: List[LineBox] = []
        for line_idx, line_word_list in enumerate(lines, start=1):
            line_text = " ".join([w.text for w in line_word_list if w.text])
            if not line_text:
                continue

            # Compute bounding box that encompasses all words in this line
            min_x = min(w.bbox.x for w in line_word_list)
            min_y = min(w.bbox.y for w in line_word_list)
            max_x = max(w.bbox.x + w.bbox.width for w in line_word_list)
            max_y = max(w.bbox.y + w.bbox.height for w in line_word_list)

            avg_conf = sum(w.confidence for w in line_word_list) / max(len(line_word_list), 1)

            line_bbox = BoundingBox.from_coords(
                min_x, min_y, max_x - min_x, max_y - min_y, img_w, img_h
            )

            line_boxes.append(LineBox(
                line_number=line_idx,
                text=line_text,
                confidence=round(avg_conf, 3),
                bbox=line_bbox,
                words=line_word_list
            ))

        return line_boxes
