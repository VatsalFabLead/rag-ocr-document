import cv2
import numpy as np
from typing import List, Tuple, Optional
from app.models.ocr_result import TableResult, TableCell, BoundingBox, WordBox

class TableService:
    """
    Step 5 — Table Extraction Service
    Detects table grids, rows, columns, and maps OCR text into tabular cells
    using morphological line kernel detection.
    """

    def detect_tables(
        self, image: np.ndarray, words: List[WordBox]
    ) -> List[TableResult]:
        """
        Detects tables from an image and associates OCR word boxes to table cells.
        """
        img_h, img_w = image.shape[:2]

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Binarize with threshold
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        # 1. Detect Horizontal Lines
        horiz_kernel_len = max(20, img_w // 40)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_len, 1))
        horiz_lines = cv2.erode(binary, horiz_kernel, iterations=2)
        horiz_lines = cv2.dilate(horiz_lines, horiz_kernel, iterations=2)

        # 2. Detect Vertical Lines
        vert_kernel_len = max(20, img_h // 40)
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_len))
        vert_lines = cv2.erode(binary, vert_kernel, iterations=2)
        vert_lines = cv2.dilate(vert_lines, vert_kernel, iterations=2)

        # 3. Combine to form table grid
        table_grid = cv2.add(horiz_lines, vert_lines)

        # 4. Find table container contours
        contours, _ = cv2.findContours(table_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tables: List[TableResult] = []
        table_counter = 1

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter out lines or tiny boxes
            if w < (img_w * 0.15) or h < 50:
                continue

            table_roi = table_grid[y:y+h, x:x+w]

            # Find cell contours within this table
            cell_contours, _ = cv2.findContours(table_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            raw_cells: List[Tuple[int, int, int, int]] = []
            for cell_cnt in cell_contours:
                cx, cy, cw, ch = cv2.boundingRect(cell_cnt)
                # Valid cell dimension
                if 15 < cw < (w * 0.95) and 10 < ch < (h * 0.95):
                    raw_cells.append((x + cx, y + cy, cw, ch))

            if not raw_cells:
                # If cells aren't fully enclosed by lines, detect rows via horizontal line segments
                continue

            # Group cells by Y coordinate into rows
            rows_cells = self._cluster_cells_into_rows(raw_cells, row_threshold=12)

            table_cells: List[TableCell] = []
            reconstructed_rows: List[List[str]] = []

            for row_idx, row in enumerate(rows_cells):
                row_texts = []
                for col_idx, (cx, cy, cw, ch) in enumerate(row):
                    cell_bbox = BoundingBox.from_coords(cx, cy, cw, ch, img_w, img_h)

                    # Find all words that fall inside this cell
                    contained_words = [
                        w for w in words
                        if self._is_box_inside((cx, cy, cw, ch), (w.bbox.x, w.bbox.y, w.bbox.width, w.bbox.height))
                    ]

                    cell_text = " ".join([w.text for w in contained_words]).strip()
                    row_texts.append(cell_text)

                    table_cells.append(TableCell(
                        row_idx=row_idx,
                        col_idx=col_idx,
                        text=cell_text,
                        confidence=0.9 if cell_text else 1.0,
                        bbox=cell_bbox
                    ))
                reconstructed_rows.append(row_texts)

            if reconstructed_rows:
                headers = reconstructed_rows[0] if len(reconstructed_rows) > 1 else []
                data_rows = reconstructed_rows[1:] if len(reconstructed_rows) > 1 else reconstructed_rows

                num_cols = max(len(r) for r in reconstructed_rows)

                tables.append(TableResult(
                    table_id=f"table_{table_counter}",
                    bbox=BoundingBox.from_coords(x, y, w, h, img_w, img_h),
                    num_rows=len(reconstructed_rows),
                    num_cols=num_cols,
                    headers=headers,
                    rows=data_rows,
                    cells=table_cells
                ))
                table_counter += 1

        return tables

    def _cluster_cells_into_rows(
        self, cells: List[Tuple[int, int, int, int]], row_threshold: int = 12
    ) -> List[List[Tuple[int, int, int, int]]]:
        cells_sorted = sorted(cells, key=lambda c: c[1])
        rows: List[List[Tuple[int, int, int, int]]] = []
        current_row: List[Tuple[int, int, int, int]] = [cells_sorted[0]]

        for cell in cells_sorted[1:]:
            prev = current_row[-1]
            if abs(cell[1] - prev[1]) <= row_threshold:
                current_row.append(cell)
            else:
                rows.append(sorted(current_row, key=lambda c: c[0]))
                current_row = [cell]

        if current_row:
            rows.append(sorted(current_row, key=lambda c: c[0]))

        return rows

    def _is_box_inside(
        self, container: Tuple[int, int, int, int], inner: Tuple[int, int, int, int]
    ) -> bool:
        cx, cy, cw, ch = container
        ix, iy, iw, ih = inner
        center_x = ix + iw / 2.0
        center_y = iy + ih / 2.0
        return (cx <= center_x <= cx + cw) and (cy <= center_y <= cy + ch)
