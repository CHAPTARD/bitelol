import sys
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtCore import Qt
from PyQt6.QtGui import *
import math
import os
from datetime import datetime
from collections import OrderedDict
import copy
import datetime
import re
        

'''color palette'''
COLORS = {
    'bg_primary': "rgba(255, 202, 161, 0.93)",
    'bg_primary_accent': "#FCA624",
    'bg_secondary': "rgba(201, 157, 255, 0.93)",
    'bg_secondary_accent': "#B06FFF",
    'glass_overlay': "#0DFF00",
    'border': "#FFFFFF",
    'text_primary': "#3C200B",
    'text_secondary': "#4D4D4D",
    'highlight': "#F4FF93",
    'search_bg': "#FFEA7FDB",
    'error': "#923964",
    'header_bg': "#007427",
    'header_text': "#0E005F",
    'table_border': "#D3B4FA",
    'selection': "rgba(118, 249, 255, 0.5)"
}

'''global functions definitions'''
def sanitize_text(text):
    import pandas as pd
    if isinstance(text, (list, tuple, dict)):
        return str(text)
    if hasattr(text, "__array__") and getattr(text, "shape", ()) != ():  # numpy array, not scalar
        return str(text)
    try:
        if pd.isna(text) or text is None:
            return ""
    except Exception:
        pass
    text_str = str(text)
    if len(text_str) > 100:
        text_str = text_str[:97] + "..."
    return text_str

def calculate_min_widths(table):
    min_widths = []
    font_metrics = table.fontMetrics()
    for col in range(table.columnCount()):
        header_item = table.horizontalHeaderItem(col)
        header_text = header_item.text() if header_item else ""
        header_width = font_metrics.horizontalAdvance(header_text) + 20
        min_width = max(header_width, 60)  # At least 60px
        min_widths.append(min_width)
    return min_widths

def level_widths(ideal_widths, min_widths, target_total):
    widths = ideal_widths.copy()
    current_total = sum(widths)
    if abs(current_total - target_total) <= 1:
        return [int(w) for w in widths]
    if current_total > target_total:
        return shrink_level_widths(widths, min_widths, target_total)
    else:
        return expand_level_widths(widths, ideal_widths, target_total)

def shrink_level_widths(widths, min_widths, target_total):
    while sum(widths) > target_total + 1:
        sorted_indices = sorted(range(len(widths)), key=lambda i: widths[i], reverse=True)
        max_width = widths[sorted_indices[0]]
        candidates = []
        for i in sorted_indices:
            if widths[i] == max_width and widths[i] > min_widths[i]:
                candidates.append(i)
            elif widths[i] < max_width:
                break
        if not candidates:
            break
        next_level = max_width
        for i in sorted_indices:
            if widths[i] < max_width:
                next_level = max(widths[i], max(min_widths[j] for j in candidates))
                break
        else:
            next_level = max(min_widths[j] for j in candidates)
        per_column_reduction = max_width - next_level
        total_possible_reduction = per_column_reduction * len(candidates)
        needed_reduction = sum(widths) - target_total
        if total_possible_reduction <= needed_reduction:
            for i in candidates:
                widths[i] = next_level
        else:
            actual_per_column = needed_reduction / len(candidates)
            for i in candidates:
                widths[i] = max(widths[i] - actual_per_column, min_widths[i])
            break
    return [int(w) for w in widths]

def expand_level_widths(current_widths, ideal_widths, target_total):
    widths = current_widths.copy()
    needed_expansion = target_total - sum(widths)
    truncated_columns = []
    for i in range(len(widths)):
        if widths[i] < ideal_widths[i]:
            truncated_columns.append(i)
    if truncated_columns:
        while needed_expansion > 1 and truncated_columns:
            truncated_columns.sort(key=lambda i: ideal_widths[i] - widths[i])
            min_gap = ideal_widths[truncated_columns[0]] - widths[truncated_columns[0]]
            candidates = []
            for i in truncated_columns:
                gap = ideal_widths[i] - widths[i]
                
                if abs(gap - min_gap) < 0.1:
                    candidates.append(i)
                elif gap > min_gap:
                    break
            total_possible_expansion = min_gap * len(candidates)
            if total_possible_expansion <= needed_expansion:
                for i in candidates:
                    widths[i] = ideal_widths[i]
                    truncated_columns.remove(i)
                needed_expansion -= total_possible_expansion
            else:
                per_column_expansion = needed_expansion / len(candidates)
                for i in candidates:
                    widths[i] += per_column_expansion
                needed_expansion = 0
                break
    if needed_expansion > 1:
        per_column_expansion = needed_expansion / len(widths)
        for i in range(len(widths)):
            widths[i] += per_column_expansion
    return [int(w) for w in widths]

def setup_text_elision(table):
    font_metrics = table.fontMetrics()
    for row in range(table.rowCount()):
        for col in range(table.columnCount()):
            item = table.item(row, col)
            if item:
                if not hasattr(item, 'original_text'):
                    item.original_text = item.text()
                column_width = table.columnWidth(col)
                available_text_width = column_width - 20
                elided_text = font_metrics.elidedText(
                    item.original_text,
                    Qt.TextElideMode.ElideRight,
                    available_text_width
                )
                item.setText(elided_text)
                if elided_text != item.original_text:
                    item.setToolTip(item.original_text)
                else:
                    item.setToolTip("")

def smart_resize_columns(table):
    if table.columnCount() == 0:
        return
    available_width = table.viewport().width()
    table.resizeColumnsToContents()
    current_widths = [table.columnWidth(i) for i in range(table.columnCount())]
    min_widths = calculate_min_widths(table)
    final_widths = level_widths(current_widths, min_widths, available_width)
    for col, width in enumerate(final_widths):
        table.setColumnWidth(col, width)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
    setup_text_elision(table)

def fix_operations_column(df):
    def fix_ops(ops):
        # Handle both list and numpy array of dicts
        if isinstance(ops, (list, tuple)):
            return [
                OrderedDict([
                    ("ordre", op.get("ordre", "")),
                    ("description", op.get("description", "")),
                    ("url", op.get("url", "")),
                ])
                for op in ops if isinstance(op, dict)
            ]
        elif hasattr(ops, "__iter__"):  # numpy array
            return [
                OrderedDict([
                    ("ordre", op.get("ordre", "")),
                    ("description", op.get("description", "")),
                    ("url", op.get("url", "")),
                ])
                for op in list(ops) if isinstance(op, dict)
            ]
        else:
            return []
    if "operations" in df.columns:
        df["operations"] = df["operations"].apply(fix_ops)

def normalize_operations_column(df):
    """
    Ensures the 'operations' column is always a list of OrderedDicts with keys in the correct order.
    """
    def normalize_ops(ops):
        # Accept only list/tuple/array-like, convert each element to OrderedDict
        if isinstance(ops, (list, tuple)):
            return [
                OrderedDict([
                    ("ordre", op.get("ordre", "")),
                    ("description", op.get("description", "")),
                    ("url", op.get("url", "")),
                ])
                for op in ops if isinstance(op, dict)
            ]
        elif hasattr(ops, "__iter__") and not isinstance(ops, (str, bytes, dict)):
            # Handles numpy arrays and similar
            return [
                OrderedDict([
                    ("ordre", op.get("ordre", "")),
                    ("description", op.get("description", "")),
                    ("url", op.get("url", "")),
                ])
                for op in list(ops) if isinstance(op, dict)
            ]
        else:
            return []
    if "operations" in df.columns:
        df["operations"] = df["operations"].apply(normalize_ops)
    return df

def export_to_excel(self):
    if self.table_manager.df is None or self.table_manager.df.empty:
        QMessageBox.warning(self, "No data", "No data to export.")
        return

    df = self.table_manager.df
    # Default directory: opened file dir or Desktop
    default_dir = os.path.dirname(self.table_manager.current_file) if self.table_manager.current_file else os.path.expanduser("~/Desktop")
    dir_path = QFileDialog.getExistingDirectory(self, "Select Export Directory", default_dir)
    if not dir_path:
        return


    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(dir_path, f"actions_unitaires_{date_str}.xlsx")

    # Prepare rows as described
    rows = []
    for _, row in df.iterrows():
        code_tache = row.get("code_tache", "")
        num_seq_op = row.get("num_seq_op", "")
        code_art_mr = row.get("code_art_mr", "")
        operations = row.get("operations", [])
        if not isinstance(operations, list):
            operations = []
        if not operations:
            rows.append([code_art_mr, num_seq_op, code_tache, "", "", ""])
        else:
            for op in operations:
                ordre = op.get("ordre", "")
                description = op.get("description", "")
                url = op.get("url", "")
                rows.append([code_art_mr, num_seq_op, code_tache, ordre, description, url])

    out_df = pd.DataFrame(rows, columns=["code_art_mr", "num_seq_op", "clef_fer", "ordre", "description", "url"])
    try:
        out_df.to_excel(out_path, index=False)
        QMessageBox.information(self, "Exported", f"Excel file saved to:\n{out_path}")
    except Exception as e:
        QMessageBox.critical(self, "Export Error", str(e))

def copy_selected_cells(table):
    """Copy selected cells to clipboard in Excel format."""
    selection = table.selectedRanges()
    if not selection:
        return
    copied = []
    for sel in selection:
        rows = []
        for row in range(sel.topRow(), sel.bottomRow() + 1):
            row_data = []
            for col in range(sel.leftColumn(), sel.rightColumn() + 1):
                item = table.item(row, col)
                row_data.append(item.text() if item else "")
            rows.append('\t'.join(row_data))
        copied.append('\n'.join(rows))
    clipboard = QApplication.clipboard()
    clipboard.setText('\n'.join(copied))

class HoverGlowButton(QPushButton):
    def __init__(self, text, parent=None, colors=None):
        super().__init__(text, parent)
        self.colors = COLORS
        self.setStyleSheet(f"""
            background: {colors['bg_secondary_accent']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border']};
            border-radius: 12px;
            padding: 6px 12px;
        """)
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(0)  # Initially no shadow
        self.shadow.setColor(Qt.GlobalColor.white)
        self.shadow.setOffset(0)
        self.setGraphicsEffect(self.shadow)

    def enterEvent(self, event):
        self.shadow.setBlurRadius(20)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.shadow.setBlurRadius(0)
        super().leaveEvent(event)

class GlassWidget(QFrame):
    closed = pyqtSignal()
    
    def __init__(self, parent, data, colors):
        super().__init__(parent)
        self.colors = COLORS
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Glassmorphic background with rounded corners
        glass = QWidget(self)
        glass.setObjectName("glass")
        glass.setStyleSheet(f"""
            QWidget#glass {{
                background: {colors['bg_secondary']};
                border-radius: 40px;
                border: 1px solid {colors['border']};
            }}
        """)
        glass_layout = QVBoxLayout(glass)
        glass_layout.setContentsMargins(40, 40, 40, 40)
        layout.addWidget(glass)

        # --- Collect summary line and tables ---
        summary_items = []
        tables = []
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                tables.append((key, value))
            else:
                # Check for empty value or NaN
                is_empty = value in [None, '', [], {}]
                is_nan = isinstance(value, float) and math.isnan(value)
                if is_empty or is_nan:
                    summary_items.append(f"{key}: <span style='color:red;font-weight:bold;'>#N/A</span>")
                else:
                    summary_items.append(f"{key}: {str(value)}")

        # Summary line at the top
        if summary_items:
            summary_label = QLabel("   |   ".join(summary_items), glass)
            summary_label.setWordWrap(True)
            summary_label.setFont(QFont('Arial', 11, QFont.Weight.Bold))
            summary_label.setTextFormat(Qt.TextFormat.RichText)
            summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            summary_label.setStyleSheet(f"""
                color: {colors['text_primary']};
                QLabel {{
                    cursor: IBeamCursor;
                }}
            """)
            glass_layout.addWidget(summary_label)

        # Tables
        self.tables = []
        for key, value in tables:
            label = QLabel(f"{key} (Table):", glass)
            label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
            label.setStyleSheet(f"color: {colors['text_primary']};")
            glass_layout.addWidget(label)

            columns = []
            seen = set()
            for d in value:
                for k in d.keys():
                    if k not in seen:
                        columns.append(k)
                        seen.add(k)
            table = QTableWidget(len(value), len(columns), glass)
            table.keyPressEvent = self._table_key_press_event(table)
            table.setHorizontalHeaderLabels(columns)
            for row_idx, row in enumerate(value):
                for col_idx, col in enumerate(columns):
                    cell_value = row.get(col, "")
                    is_empty = cell_value in [None, '', [], {}]
                    is_nan = isinstance(cell_value, float) and math.isnan(cell_value)
                    if col.lower() == "url" and isinstance(cell_value, str) and cell_value.strip():
                        label = QLabel()
                        label.setText(f'<a href="{cell_value}">{cell_value}</a>')
                        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
                        label.setOpenExternalLinks(True)
                        label.setStyleSheet("color: #0074D9; text-decoration: underline;")
                        table.setCellWidget(row_idx, col_idx, label)
                        item = QTableWidgetItem(cell_value)
                        table.setItem(row_idx, col_idx, item)
                    elif is_empty or is_nan:
                        item = QTableWidgetItem("#N/A")
                        item.setToolTip("#N/A")
                        item.setForeground(QColor("red"))
                        table.setItem(row_idx, col_idx, item)
                    else:
                        item = QTableWidgetItem(sanitize_text(cell_value))
                        table.setItem(row_idx, col_idx, item)
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
            table.setStyleSheet(f"""
                QTableWidget {{
                    background: {self.colors['bg_primary']};
                    color: {self.colors['text_primary']};
                    border: 1px solid {self.colors['border']};
                }}
                QTableWidget::item:selected {{
                    background: {self.colors['selection']};
                    color: #000000;
                }}
                QHeaderView::section {{
                    background: {self.colors['bg_primary_accent']};
                    color: {self.colors['text_primary']};
                    border: 1px solid {self.colors['border']};
                }}
            """)

            # Wrap in a QFrame for visual effect
            frame = QFrame(glass)
            frame.setStyleSheet(f"""
                QFrame {{
                    background: {self.colors['bg_primary_accent']};
                    border: none;
                }}
            """)
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(0, 0, 0, 0)
            frame_layout.addWidget(table)
            glass_layout.addWidget(frame)
            self.tables.append(table)

    def close_widget(self):
        self.closed.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self.search_widget:
                self.close_search_overlay()
            elif self.glass_widget:
                self.close_glass_overlay()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.resize_tables)

    def resize_tables(self):
        for table in getattr(self, "tables", []):
            smart_resize_columns(table)

    def _table_key_press_event(self, table):
        orig = table.keyPressEvent
        def handler(event):
            if event.matches(QKeySequence.StandardKey.Copy):
                copy_selected_cells(table)
            else:
                orig(event)
        return handler

class TableManager:
    """
    Handles all logic related to the table: loading/parsing data, filtering, populating,
    and managing item data. Does not handle UI window behaviour.
    """
    def __init__(self, table_widget, status_label, row_count_label, file_label, colors):
        self.table = table_widget
        self.status_label = status_label
        self.row_count_label = row_count_label
        self.file_label = file_label
        self.colors = COLORS
        self.df = None
        self.filtered_df = None
        self.current_file = None
        self.filters = {}
        self.item_data = {}

    def load_parquet_file(self, file_path):
        try:
            try:
                table = pq.read_table(file_path)
                self.df = table.to_pandas()
            except Exception:
                try:
                    table = pq.read_table(file_path, use_threads=False)
                    self.df = table.to_pandas(strings_to_categorical=False, ignore_metadata=True)
                except Exception:
                    try:
                        self.df = pd.read_parquet(file_path, engine='pyarrow')
                    except Exception:
                        parquet_file = pq.ParquetFile(file_path)
                        table = parquet_file.read()
                        self.df = table.to_pandas(strings_to_categorical=False)
                        for col in self.df.columns:
                            self.df[col] = self.df[col].astype('object')

            # Always normalize operations column after loading
            self.df = normalize_operations_column(self.df)

            self.filtered_df = self.df.copy()
            self.current_file = file_path
            self.update_file_info()
            self.populate_table()
            self.status_label.setText("File loaded successfully")

        except Exception as e:
            raise Exception(f"Error loading parquet file: {str(e)}")

    def update_file_info(self):
        if self.current_file:
            filename = self.current_file.split('/')[-1].split('\\')[-1]
            rows, cols = self.df.shape
            self.file_label.setText(f"{filename} ({rows} rows, {cols} columns)")
            self.row_count_label.setText(f"Showing {len(self.filtered_df)} of {len(self.df)} rows")

    def show_empty_message(self, parent):
        # Hide table and show a message label with padding
        self.table.hide()
        if not hasattr(parent, "empty_label"):
            parent.empty_label = QLabel("Please open a parquet file", parent.centralWidget())
            parent.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            parent.empty_label.setStyleSheet(f"""
                background: {self.colors['bg_primary']};
                color: {self.colors['text_primary']};
                border-radius: 24px;
                padding: 50px;
                font-size: 28px;
                font-weight: bold;
            """)
            parent.centralWidget().layout().insertWidget(1, parent.empty_label, stretch=1)
        parent.empty_label.show()

    def populate_table(self, parent=None):
        self.table.clear()
        if self.filtered_df is None or self.filtered_df.empty:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            if parent:
                self.show_empty_message(parent)
            return
        if parent and hasattr(parent, "empty_label"):
            parent.empty_label.hide()
        self.table.show()

        uid_col = self.detect_uid_column(self.filtered_df)
        columns = list(self.filtered_df.columns)
        display_columns = columns.copy()
        if uid_col:
            display_columns.remove(uid_col)
            display_columns = ["UID"] + display_columns

        self.table.setColumnCount(len(display_columns))
        self.table.setHorizontalHeaderLabels(display_columns)
        self.table.setRowCount(len(self.filtered_df))

        self.item_data = {}
        for row_idx, (df_index, row) in enumerate(self.filtered_df.iterrows()):
            values = []
            if uid_col:
                values.append(sanitize_text(row[uid_col]))
            for col in columns:
                if col == uid_col:
                    continue
                cell_value = row[col]
                if col == "operations":
                    if isinstance(cell_value, list):
                        operations_str = "; ".join(
                            f"{a.get('ordre', '')}: {a.get('description', '')}" for a in cell_value
                        )
                        values.append(operations_str)
                    else:
                        values.append(str(cell_value))
                else:
                    values.append(sanitize_text(cell_value))
            for col_idx, value in enumerate(values):
                if value in [None, '', [], {}]:
                    item = QTableWidgetItem("#N/A")
                    item.setToolTip("#N/A")
                    item.setForeground(QColor("red"))
                else:
                    item = QTableWidgetItem(value)
                # Store DataFrame index in the first column's item for mapping after sorting
                if col_idx == 0:
                    item.setData(Qt.ItemDataRole.UserRole, df_index)
                self.table.setItem(row_idx, col_idx, item)
            self.item_data[row_idx] = row.to_dict()

        self.table.setSortingEnabled(True)
        smart_resize_columns(self.table)

    def detect_uid_column(self, df):
        preferred_names = ["uid", "id", "Clef Fer", "primary_key"]
        for name in preferred_names:
            if name in df.columns:
                try:
                    if df[name].apply(lambda x: isinstance(x, (str, int, float, bool, type(None)))).all():
                        if df[name].is_unique and df[name].notnull().all():
                            return name
                except Exception:
                    continue
        for col in df.columns:
            try:
                if df[col].apply(lambda x: isinstance(x, (str, int, float, bool, type(None)))).all():
                    if df[col].is_unique and df[col].notnull().all():
                        return col
            except Exception:
                continue
        return None

    def apply_filters(self):
        if not self.filters:
            self.filtered_df = self.df.copy()
        else:
            filtered = self.df.copy()
            for filter_data in self.filters.values():
                try:
                    if filter_data['operation'] in ["empty", "!empty"]:
                        filter_data['value'] = ""  # ensures value exists
                    filtered = self.apply_single_filter(filtered, filter_data)
                except Exception as e:
                    QMessageBox.critical(self.table, "Filter Error", f"Error applying filter: {str(e)}")
                    continue
            self.filtered_df = filtered
        self.populate_table()
        self.update_file_info()
        self.update_filter_status()

    def update_filter_status(self):
        if not self.filters:
            self.status_label.setText("No active filters")
        else:
            filter_texts = [
                f"{f['column']} {f['operation']} '{f['value']}'"
                for f in self.filters.values()
            ]
            self.status_label.setText("Filters: " + " | ".join(filter_texts))

    def apply_single_filter(self, df, filter_data):
        column = filter_data['column']
        operation = filter_data['operation']
        value = filter_data['value']
        case_sensitive = filter_data.get('case_sensitive', False)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        col_data = df[column].astype(str)
        # For date operations, parse as datetime
        def parse_euro_datetime(user_input, default="01/01/2000 00:00:00.000"):
            """
            Parse a European-format datetime string, auto-filling missing parts from the default.
            Returns a pandas.Timestamp or raises ValueError.
            """
            # Remove whitespace
            user_input = user_input.strip()
            if not user_input:
                user_input = default

            # Split date and time
            if " " in user_input:
                date_part, time_part = user_input.split(" ", 1)
            else:
                date_part, time_part = user_input, ""

            # Parse date
            date_match = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", date_part)
            if not date_match:
                raise ValueError("Date must be in format DD/MM/YYYY")
            day, month, year = date_match.groups()

            # Parse time
            time_match = re.match(r"^(\d{2}):(\d{2}):(\d{2})(\.\d{1,3})?$", time_part) if time_part else None
            if time_match:
                hour, minute, second, ms = time_match.groups()
                ms = ms if ms else ".000"
            elif not time_part:
                hour, minute, second, ms = "00", "00", "00", ".000"
            else:
                raise ValueError("Time must be in format HH:MM:SS(.mmm)")

            # Build full string
            full_str = f"{day}/{month}/{year} {hour}:{minute}:{second}{ms}"
            try:
                return pd.to_datetime(full_str, format="%d/%m/%Y %H:%M:%S.%f")
            except Exception:
                raise ValueError("Date/time format incorrect. Use DD/MM/YYYY HH:MM:SS.mmm")

        if operation == "contains":
            mask = col_data.str.contains(value, case=case_sensitive, na=False)
        elif operation == "does_not_contain":
            mask = ~col_data.str.contains(value, case=case_sensitive, na=False)
        elif operation == "equals":
            mask = col_data.str.lower() == value.lower() if not case_sensitive else col_data == value
        elif operation == "starts_with":
            mask = col_data.str.startswith(value, na=False) if case_sensitive else col_data.str.lower().str.startswith(value.lower(), na=False)
        elif operation == "ends_with":
            mask = col_data.str.endswith(value, na=False) if case_sensitive else col_data.str.lower().str.endswith(value.lower(), na=False)
        elif operation == "greater_than":
            try:
                mask = pd.to_numeric(col_data, errors='coerce') > float(value)
            except ValueError:
                raise ValueError("Greater than comparison requires numeric value")
        elif operation == "less_than":
            try:
                mask = pd.to_numeric(col_data, errors='coerce') < float(value)
            except ValueError:
                raise ValueError("Less than comparison requires numeric value")
        elif operation == "empty":
            mask = (col_data == "") | (col_data.isna()) | (col_data.str.lower() == "nan")
        elif operation == "!empty":
            mask = ~( (col_data == "") | (col_data.isna()) | (col_data.str.lower() == "nan") )
        elif operation == "date_before":
            try:
                date_val = parse_euro_datetime(value)
            except ValueError as e:
                raise ValueError(f"Invalid date for 'before': {e}")
            mask = pd.to_datetime(df[column], format="%d/%m/%Y %H:%M:%S.%f", errors='coerce') < date_val
        elif operation == "date_after":
            try:
                date_val = parse_euro_datetime(value)
            except ValueError as e:
                raise ValueError(f"Invalid date for 'after': {e}")
            mask = pd.to_datetime(df[column], format="%d/%m/%Y %H:%M:%S.%f", errors='coerce') > date_val
        elif operation == "date_between":
            try:
                parts = [v.strip() for v in value.split(";")]
                if len(parts) != 2:
                    raise ValueError("For 'date_between', enter two dates separated by ';'")
                start = parse_euro_datetime(parts[0])
                end = parse_euro_datetime(parts[1])
            except ValueError as e:
                raise ValueError(f"Invalid date for 'between': {e}")
            col_dates = pd.to_datetime(df[column], format="%d/%m/%Y %H:%M:%S.%f", errors='coerce')
            mask = (col_dates >= start) & (col_dates <= end)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        return df[mask]

class BlurryBackground(QWidget):
    def __init__(self, parent, colors, radius):
        super().__init__(parent)
        self.colors = COLORS
        self.radius = radius
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet(f"""
            background: rgba(255, 255, 255, 0.35);
            border-radius: {radius}px;
        """)
        blur = QGraphicsBlurEffect(self)
        blur.setBlurRadius(32)
        self.setGraphicsEffect(blur)
        self.setGeometry(self.parent().rect())

class BlurryFrame(QFrame):
    """A QFrame with frosted glass background blur and sharp outline."""
    def __init__(self, parent, colors, outline_width=1, radius=32):
        super().__init__(parent)
        self.colors = COLORS
        self.outline_width = outline_width
        self.radius = radius
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        
        # Add blurry background widget as a child
        self.blur_bg = BlurryBackground(self, colors, radius)
        self.blur_bg.lower()  # make sure background is behind

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.blur_bg.setGeometry(self.rect())  # keep blur background sized

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(
            self.outline_width // 2,
            self.outline_width // 2,
            -self.outline_width // 2,
            -self.outline_width // 2
        )
        pen = QPen(QColor(self.colors['border']), self.outline_width)
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawRoundedRect(rect, self.radius, self.radius)

class DragDropTableWidget(QTableWidget):
    """Custom QTableWidget with drag and drop functionality for row reordering"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        
    def dropEvent(self, event):
        """Handle drop events for row reordering"""
        if event.source() != self:
            return
        
        drop_row = self.indexAt(event.position().toPoint()).row()
        if drop_row == -1:
            drop_row = self.rowCount()
        
        selected_rows = sorted(set(item.row() for item in self.selectedItems()))
        
        if not selected_rows:
            return
        
        # Store the data from selected rows
        rows_data = []
        for row in selected_rows:
            row_data = []
            for col in range(self.columnCount()):
                item = self.item(row, col)
                row_data.append(item.text() if item else "")
            rows_data.append(row_data)
        
        # Remove selected rows (in reverse order to maintain indices)
        for row in reversed(selected_rows):
            self.removeRow(row)
            if row < drop_row:
                drop_row -= 1
        
        # Insert rows at drop position
        for i, row_data in enumerate(rows_data):
            self.insertRow(drop_row + i)
            for col, data in enumerate(row_data):
                if col == 0:  # Skip ordre column, it will be updated automatically
                    continue
                item = QTableWidgetItem(data)
                self.setItem(drop_row + i, col, item)
        
        # Update ordre column for all rows
        self.update_ordre_column()
        
        # Select the moved rows
        self.clearSelection()
        for i in range(len(rows_data)):
            for col in range(self.columnCount()):
                self.item(drop_row + i, col).setSelected(True)
        
        event.accept()
    
    def update_ordre_column(self):
        """Update the ordre column with sequential numbers"""
        for row in range(self.rowCount()):
            item = self.item(row, 0)
            if not item:
                item = QTableWidgetItem()
                self.setItem(row, 0, item)
            item.setText(str(row + 1))
            # Make ordre column non-editable
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

class operationsListEditor(QFrame):
    closed = pyqtSignal(bool)

    def __init__(self, parent, items, colors):
        super().__init__(parent)
        self.colors = colors
        self.items = items
        self.edit_all = False
        self.clipboard_data = []

        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            operationsListEditor {{
                background: {colors['bg_secondary']};
                border-radius: 24px;
                border: 1px solid {colors['border']};
            }}
        """)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        self.tables = []

        if len(items) > 1:
            # Tab widget at the top
            self.tabs = QTabWidget(self)
            self.tabs.setStyleSheet(f"""
                QTabWidget::pane {{
                    border: 1px solid white;
                    background: {colors['bg_primary']};
                }}
                QTabBar::tab {{
                    background: {colors['bg_primary']};
                    color: {colors['text_primary']};
                    border: 1px solid {colors['border']};
                    padding: 8px 16px;
                    margin-right: 2px;
                }}
                QTabBar::tab:selected {{
                    background: {colors['bg_primary_accent']};
                }}
            """)
            self.tabs.currentChanged.connect(self.on_tab_changed)
            layout.addWidget(self.tabs)

            # Create tables for each item
            for i, item in enumerate(items):
                tab = QWidget()
                vbox = QVBoxLayout(tab)
                vbox.setContentsMargins(0, 0, 0, 0)
                vbox.setSpacing(0)

                table = self.create_table()
                self.setup_table_data(table, item.get('operations', []))
                vbox.addWidget(table)

                tab_name = item.get('code_tache', f'List {i+1}')
                self.tabs.addTab(tab, tab_name)
                self.tables.append(table)
                smart_resize_columns(table)

            # All-table for "edit all" mode (hidden by default)
            self.all_table = self.create_table()
            self.all_table.hide()
            layout.addWidget(self.all_table)
            # --- Connect itemChanged to live-update all tables ---
            self.all_table.itemChanged.connect(self.on_all_table_changed)
        else:
            # Only one item: no tabs, just one table
            table = self.create_table()
            self.setup_table_data(table, items[0].get('operations', []))
            layout.addWidget(table)
            self.tables.append(table)
            smart_resize_columns(table)
            self.all_table = None  # No all-table needed

        # Buttons layout
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        # Toggle edit all button (only if multiple items)
        if len(items) > 1:
            self.toggle_btn = self.create_button("Edit All Selected Lists", colors)
            self.toggle_btn.setCheckable(True)
            self.toggle_btn.clicked.connect(self.toggle_mode)
            buttons_layout.addWidget(self.toggle_btn)
        else:
            self.toggle_btn = None

        # Add row button
        self.add_btn = self.create_button("Add Row", colors)
        self.add_btn.clicked.connect(self.add_row_to_current_table)
        buttons_layout.addWidget(self.add_btn)

        buttons_layout.addStretch()

        # OK button
        ok_btn = self.create_button("OK", colors)
        ok_btn.clicked.connect(lambda: self.close_and_emit(True))
        buttons_layout.addWidget(ok_btn)

        # Cancel button
        cancel_btn = self.create_button("Cancel", colors)
        cancel_btn.clicked.connect(lambda: self.close_and_emit(False))
        buttons_layout.addWidget(cancel_btn)

        layout.addLayout(buttons_layout)

    def create_button(self, text, colors):
        """Create a styled button with white border"""
        btn = QPushButton(text, self)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {colors['bg_primary']};
                color: {colors['text_primary']};
                border: 1px solid {colors['border']};
                border-radius: 16px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {colors['bg_primary_accent']};
            }}
            QPushButton:pressed {{
                background: {colors.get('bg_primary_pressed', colors['bg_primary_accent'])};
            }}
            QPushButton:checked {{
                background: {colors['bg_primary_accent']};
            }}
        """)
        return btn

    def create_table(self):
        """Create a table widget with proper styling and Excel-like functionality"""
        table = DragDropTableWidget(self)
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["ordre", "description", "url"])
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)  # Changed to SelectRows for better drag experience
        table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        
        # Enable context menu for right-click operations
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(lambda pos, t=table: self.show_context_menu(pos, t))
        
        table.setStyleSheet(f"""
            QTableWidget {{
                background: {self.colors['bg_primary']};
                color: {self.colors['text_primary']};
                border: 1px solid white;
                gridline-color: {self.colors['border']};
            }}
            QHeaderView::section {{
                background: {self.colors['bg_primary_accent']};
                color: {self.colors['text_primary']};
                border: 1px solid white;
                padding: 5px;
            }}
            QTableWidget::item:selected {{
                background: {self.colors.get('selection_bg', '#3daee9')};
            }}
        """)
        
        # Install event filter for keyboard shortcuts
        table.installEventFilter(self)
        
        return table

    def setup_table_data(self, table, operations):
        """Setup table with operation data. URLs are plain text (not clickable) here."""
        table.setRowCount(len(operations))
        for r, act in enumerate(operations):
            ordre = str(act.get('ordre', r+1))
            desc = str(act.get('description', ''))
            url = str(act.get('url', ''))
            for c, val in enumerate([ordre, desc, url]):
                display_val = val if val not in [None, '', [], {}] else ""
                item = QTableWidgetItem(display_val)
                if display_val == "" and c != 0:
                    item.setText("#N/A")
                    item.setForeground(QColor("red"))
                    item.setToolTip("#N/A")
                elif len(display_val) > 100:
                    item.setToolTip(display_val)
                # Make ordre column non-editable
                if c == 0:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(r, c, item)
        smart_resize_columns(table)

    def show_context_menu(self, pos, table):
        """Show context menu for copy/cut/paste operations"""
        menu = QMenu(self)
        
        copy_action = menu.addAction("Copy")
        cut_action = menu.addAction("Cut")
        paste_action = menu.addAction("Paste")
        menu.addSeparator()
        delete_action = menu.addAction("Delete Row(s)")
        
        copy_action.triggered.connect(lambda: self.copy_selection(table))
        cut_action.triggered.connect(lambda: self.cut_selection(table))
        paste_action.triggered.connect(lambda: self.paste_selection(table))
        delete_action.triggered.connect(lambda: self.delete_selected_rows(table))
        
        # Enable/disable paste based on clipboard content
        paste_action.setEnabled(len(self.clipboard_data) > 0)
        
        menu.exec(table.mapToGlobal(pos))

    def delete_selected_rows(self, table):
        """Delete selected rows and update ordre column"""
        selected_rows = sorted(set(item.row() for item in table.selectedItems()), reverse=True)
        for row in selected_rows:
            table.removeRow(row)
        table.update_ordre_column()

    def eventFilter(self, obj, event):
        """Handle keyboard shortcuts for copy/cut/paste"""
        if event.type() == QEvent.Type.KeyPress and isinstance(obj, QTableWidget):
            if event.matches(QKeySequence.StandardKey.Copy):
                self.copy_selection(obj)
                return True
            elif event.matches(QKeySequence.StandardKey.Cut):
                self.cut_selection(obj)
                return True
            elif event.matches(QKeySequence.StandardKey.Paste):
                self.paste_selection(obj)
                return True
            elif event.key() == Qt.Key.Key_Delete:
                self.delete_selected_rows(obj)
                return True
        return super().eventFilter(obj, event)

    def copy_selection(self, table):
        """Copy selected cells to internal clipboard"""
        selection = table.selectedItems()
        if not selection:
            return
        
        # Get selection bounds
        rows = set()
        cols = set()
        for item in selection:
            rows.add(item.row())
            cols.add(item.column())
        
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        
        # Store data
        self.clipboard_data = []
        for r in range(min_row, max_row + 1):
            row_data = []
            for c in range(min_col, max_col + 1):
                item = table.item(r, c)
                row_data.append(item.text() if item else "")
            self.clipboard_data.append(row_data)

    def cut_selection(self, table):
        """Cut selected cells (copy + clear)"""
        self.copy_selection(table)
        selection = table.selectedItems()
        for item in selection:
            if item.column() != 0:  # Don't clear ordre column
                item.setText("")

    def paste_selection(self, table):
        if not self.clipboard_data:
            return
    
        selection = table.selectedItems()
        if not selection:
            return
    
        start_row = min(item.row() for item in selection)
        start_col = min(item.column() for item in selection)
        if start_col == 0:
            start_col = 1
    
        needed_rows = start_row + len(self.clipboard_data)
        if needed_rows > table.rowCount():
            table.setRowCount(needed_rows)
            table.update_ordre_column()
    
        for r, row_data in enumerate(self.clipboard_data):
            target_row = start_row + r
            for c, cell_data in enumerate(row_data):
                source_col = c
                target_col = start_col + c
                if len(row_data) == 3 and start_col <= 1:
                    if source_col == 0:
                        continue
                    elif source_col == 1:
                        target_col = 1
                    elif source_col == 2:
                        target_col = 2
                else:
                    target_col = start_col + source_col
                if target_col == 0 or target_col >= table.columnCount():
                    continue
    
                item = table.item(target_row, target_col)
                if not item:
                    item = QTableWidgetItem("")
                    table.setItem(target_row, target_col, item)
                # Only set red N/A for display, not as data
                if cell_data in [None, '', [], {}]:
                    item.setText("#N/A")
                    item.setToolTip("#N/A")
                    item.setForeground(QColor("red"))
                else:
                    item.setText(cell_data)
                    item.setToolTip("")
                    item.setForeground(QColor("black"))
        smart_resize_columns(table)

    def get_current_table(self):
        """Get the currently active table"""
        if self.edit_all and self.all_table and self.all_table.isVisible():
            return self.all_table
        elif hasattr(self, "tabs") and self.tabs is not None:
            current_index = self.tabs.currentIndex()
            return self.tables[current_index] if 0 <= current_index < len(self.tables) else None
        elif self.tables:
            return self.tables[0]
        return None

    def add_row_to_current_table(self):
        """Add row to currently active table"""
        table = self.get_current_table()
        if table:
            self.add_row(table)

    def apply_batch_edits(self):
        """Apply the current batch table to all individual tables and items."""
        if not (self.edit_all and self.all_table and self.all_table.isVisible()):
            return
        ops = []
        for r in range(self.all_table.rowCount()):
            desc_item = self.all_table.item(r, 1)
            url_item = self.all_table.item(r, 2)
            desc = desc_item.text() if desc_item else ""
            if desc == "#N/A":
                desc = ""
            url = url_item.text() if url_item else ""
            if url == "#N/A":
                url = ""
            if desc or url:
                ops.append(OrderedDict([
                    ("ordre", r + 1),
                    ("description", desc),
                    ("url", url)
                ]))
        # Apply to all tables and items
        for i, table in enumerate(self.tables):
            self.setup_table_data(table, ops)
        for i, item in enumerate(self.items):
            item['operations'] = copy.deepcopy(ops)

    def toggle_mode(self):
        self.edit_all = self.toggle_btn.isChecked()
        self.tabs.setVisible(not self.edit_all)
    
        if self.edit_all:
            # Switching to batch mode: copy data from first table to all_table
            if self.tables:
                first_table = self.tables[0]
                self.all_table.setRowCount(first_table.rowCount())
                for r in range(first_table.rowCount()):
                    for c in range(3):
                        item = first_table.item(r, c)
                        new_item = QTableWidgetItem(item.text() if item else "")
                        if c == 0:  # Make ordre column non-editable
                            new_item.setFlags(new_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        self.all_table.setItem(r, c, new_item)
            self.all_table.show()
            smart_resize_columns(self.all_table)
            self.toggle_btn.setText("Edit Individual Lists")
        else:
            # Switching to individual mode: apply batch edits before hiding all_table
            self.apply_batch_edits()  # <-- This copies global table to individual tables
            for i, table in enumerate(self.tables):
                print(f"Table {i}:")
                for r in range(table.rowCount()):
                    print([table.item(r, c).text() if table.item(r, c) else "" for c in range(3)])
            self.all_table.hide()
            for table in self.tables:
                smart_resize_columns(table)
            self.toggle_btn.setText("Edit All Selected Lists")

    def add_row(self, table):
        row = table.rowCount()
        table.insertRow(row)
        ordre_item = QTableWidgetItem(str(row + 1))
        ordre_item.setFlags(ordre_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table.setItem(row, 0, ordre_item)
        for col in [1, 2]:
            item = QTableWidgetItem("")
            item.setText("#N/A")
            item.setToolTip("#N/A")
            item.setForeground(QColor("red"))
            table.setItem(row, col, item)
        smart_resize_columns(table)

    def build_row_dict(r, table):
        ordre = r + 1
        desc_item = table.item(r, 1)
        url_item = table.item(r, 2)
        desc = desc_item.text() if desc_item else ""
        url = url_item.text() if url_item else ""
        if desc or url:
            # Use OrderedDict to enforce key order
            return OrderedDict([("ordre", ordre), ("description", desc), ("url", url)])
        else:
            return None
            
        if self.edit_all and self.all_table and self.all_table.isVisible():
            # Edit all mode: apply same operations to all items
            operations = []
            for r in range(self.all_table.rowCount()):
                row_dict = build_row_dict(r, self.all_table)
                if row_dict:
                    operations.append(row_dict)
            
            # Return the same operations list for each item
            result = []
            for _ in self.items:
                result.append(copy.deepcopy(operations))
            return result
        else:
            # Individual mode: get operations from each table
            result = []
            
            # Handle single item case properly
            if len(self.items) == 1:
                # Single item - return list with one operations list
                operations = []
                table = self.tables[0]
                for r in range(table.rowCount()):
                    row_dict = build_row_dict(r, table)
                    if row_dict:
                        operations.append(row_dict)
                result.append(operations)
            else:
                # Multiple items - return list of operations lists
                for table in self.tables:
                    operations = []
                    for r in range(table.rowCount()):
                        row_dict = build_row_dict(r, table)
                        if row_dict:
                            operations.append(row_dict)
                    result.append(operations)
            
            return result

    def close_and_emit(self, accepted):
        """Close the widget and emit the result"""
        # --- FIX: If in batch mode, apply batch edits before closing ---
        if accepted and self.edit_all and self.all_table and self.all_table.isVisible():
            self.apply_batch_edits()
        self.hide()
        self.closed.emit(accepted)

    def keyPressEvent(self, event):
        """Handle key press events including Ctrl+PgUp and Ctrl+PgDown to switch tabs"""
        if event.key() == Qt.Key.Key_Escape:
            self.closed.emit(False)
            return
        
        if hasattr(self, 'tabs') and self.tabs is not None and self.tabs.count() > 1:
            if event.key() == Qt.Key.Key_PageUp and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                current_index = self.tabs.currentIndex()
                new_index = (current_index + 1) % self.tabs.count()
                self.tabs.setCurrentIndex(new_index)
                event.accept()
                return
            elif event.key() == Qt.Key.Key_PageDown and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                current_index = self.tabs.currentIndex()
                new_index = (current_index - 1) % self.tabs.count()
                self.tabs.setCurrentIndex(new_index)
                event.accept()
                return
        
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if not self.rect().contains(event.pos()):
            self.closed.emit(False)
        else:
            super().mousePressEvent(event)

    def close_widget(self):
        """Close the widget"""
        self.closed.emit(False)

    def get_updated_lists(self):
        """
        Returns a list of lists, each containing OrderedDicts for the operations of each edited row.
        The order matches the order of the items/rows passed to the editor.
        """
        results = []
        if self.edit_all and self.all_table and self.all_table.isVisible():
            table = self.all_table
            ops = []
            for r in range(table.rowCount()):
                desc_item = table.item(r, 1)
                url_item = table.item(r, 2)
                desc = desc_item.text() if desc_item else ""
                if desc == "#N/A":
                    desc = ""
                url = url_item.text() if url_item else ""
                if url == "#N/A":
                    url = ""
                if desc or url:
                    ops.append(OrderedDict([
                        ("ordre", r + 1),
                        ("description", desc),
                        ("url", url)
                    ]))
            results = [copy.deepcopy(ops) for _ in self.items]
        elif hasattr(self, "tables") and self.tables:
            for i, table in enumerate(self.tables):
                ops = []
                for r in range(table.rowCount()):
                    desc_item = table.item(r, 1)
                    url_item = table.item(r, 2)
                    desc = desc_item.text() if desc_item else ""
                    if desc == "#N/A":
                        desc = ""
                    url = url_item.text() if url_item else ""
                    if url == "#N/A":
                        url = ""
                    if desc or url:
                        ops.append(OrderedDict([
                            ("ordre", r + 1),
                            ("description", desc),
                            ("url", url)
                        ]))
                results.append(ops)
        return results

    def on_all_table_changed(self, item):
        """Whenever the batch table changes, update all individual tables immediately."""
        if self.edit_all and self.all_table.isVisible():
            self.apply_batch_edits()

    def showEvent(self, event):
        """Called when the widget is shown - resize all table columns here"""
        super().showEvent(event)
        # Resize all individual tables
        for table in self.tables:
            smart_resize_columns(table)
        # Resize all-table if it exists and is visible
        if hasattr(self, 'all_table') and self.all_table and self.all_table.isVisible():
            smart_resize_columns(self.all_table)

    def on_tab_changed(self, index):
        """Called when tab is changed - resize columns of the newly visible table"""
        if 0 <= index < len(self.tables):
            # Use QTimer.singleShot to ensure the tab switch is complete before resizing
            QTimer.singleShot(0, lambda: smart_resize_columns(self.tables[index]))

class ParquetViewer(QMainWindow):
    BORDER_WIDTH = 8
    OUTLINE_WIDTH = 1
    export_to_excel = export_to_excel  # Attach as method
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Universal Parquet Viewer")
        self.resize(1400, 800)
        self._startup_geometry = QRect(100, 100, 1400, 800)  # Default startup geometry
        self.setGeometry(self._startup_geometry)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self._resizing = False
        self._resize_dir = None
        self._mouse_press_pos = None
        self._mouse_press_geom = None
        self._dragging = False
        self._drag_start_pos = None
        self._drag_start_geom = None
        self.colors = COLORS
        self.settings = QSettings("Providence", "ParquetViewer")
        self.init_ui()
        self.centralWidget().installEventFilter(self)
        self.search_widget = None
        self.overlay = None
        self.glass_widget = None
        self.glass_overlay = None
        self.screen_geometry = QApplication.primaryScreen().availableGeometry()
        self.current_glass_row = None
        self.filtered_row_indices = []
        self.undo_stack = []
        self.redo_stack = []

        # TableManager instance (must be after self.table is created)
        # This block is moved to the end of init_ui
        # TableManager instance (must be after self.table is created)
        self.table_manager = TableManager(
            self.table, self.status_label, self.row_count_label, self.file_label, self.colors
        )
        #self.table_manager.df.to_parquet(self.table_manager.current_file)

        # Try to reopen last file
        last_file = self.settings.value("last_file", "")
        if last_file and os.path.exists(last_file):
            try:
                self.table_manager.load_parquet_file(last_file)
            except Exception:
                self.table_manager.show_empty_message(self)
        else:
            self.table_manager.show_empty_message(self)

        self.dirty = False  # Track unsaved changes

        # Shortcuts
        self.shortcut_search = QShortcut(QKeySequence("Ctrl+F"), self)
        self.shortcut_search.activated.connect(self.open_search)
        self.shortcut_open = QShortcut(QKeySequence("Ctrl+O"), self)
        self.shortcut_open.activated.connect(self.open_file)
        self.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save.activated.connect(self.save_file)
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.undo)
        self.shortcut_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.shortcut_redo.activated.connect(self.redo)
        self.shortcut_edit = QShortcut(QKeySequence("Ctrl+E"), self)
        self.shortcut_edit.activated.connect(self.edit_selected_rows)

    def init_ui(self):
        # --- Add this block for the semi-transparent white QFrame ---
        self.underlay_frame = QFrame(self)
        self.underlay_frame.setGeometry(self.rect())
        self.underlay_frame.setStyleSheet("""
            background: rgba(255, 255, 255, 50);
            border-radius: 32px;
            border: none;
        """)
        self.underlay_frame.lower()  # Send to back

        # Blurry background as a sibling, not parent
        self.bg_frame = BlurryFrame(self, self.colors, outline_width=self.OUTLINE_WIDTH, radius=32)
        self.bg_frame.setGeometry(self.rect())
        self.bg_frame.lower()  # Always behind all widgets

        # Main widget is a direct child of self (the window)
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_widget.setMouseTracking(True)
        self.setMouseTracking(True)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Control panel
        self.control_panel = QFrame(main_widget)
        self.control_panel.setStyleSheet(f"background: {self.colors['bg_secondary']}; border: 1px solid {self.colors['border']}; border-radius: 24px; padding: 6px 6px;")
        control_layout = QHBoxLayout(self.control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)

        open_btn = HoverGlowButton("Open Parquet File (Ctrl+O)", self.control_panel, self.colors)
        open_btn.clicked.connect(self.open_file)
        control_layout.addWidget(open_btn)

        self.file_label = QLabel("No file loaded", self.control_panel)
        self.file_label.setStyleSheet(f"background: transparent; color: {self.colors['text_secondary']}; border : none;")
        control_layout.addWidget(self.file_label)

        control_layout.addStretch()

        search_btn = HoverGlowButton("Search (Ctrl+F)", self.control_panel, self.colors)
        search_btn.clicked.connect(self.open_search)
        control_layout.addWidget(search_btn)

        edit_btn = HoverGlowButton("Edit Selected", self.control_panel, self.colors)
        edit_btn.clicked.connect(self.edit_selected_rows)
        control_layout.addWidget(edit_btn)

        create_btn = HoverGlowButton("Create task", self.control_panel, self.colors)
        create_btn.clicked.connect(self.create_new_row)
        control_layout.addWidget(create_btn)

        delete_btn = HoverGlowButton("Delete task", self.control_panel, self.colors)
        delete_btn.clicked.connect(self.delete_selected_rows)
        control_layout.addWidget(delete_btn)

        export_btn = HoverGlowButton("💾", self.control_panel, self.colors)
        export_btn.setFixedWidth(36)
        export_btn.clicked.connect(self.export_to_excel)
        control_layout.addWidget(export_btn)

        minimize_btn = HoverGlowButton("—", self.control_panel, self.colors)
        minimize_btn.setFixedWidth(36)
        minimize_btn.clicked.connect(self.showMinimized)
        control_layout.addWidget(minimize_btn)

        maximize_btn = HoverGlowButton("□", self.control_panel, self.colors)
        maximize_btn.setFixedWidth(36)
        maximize_btn.clicked.connect(self.toggle_max_restore)
        control_layout.addWidget(maximize_btn)

        close_btn = HoverGlowButton("✕", self.control_panel, self.colors)
        close_btn.setFixedWidth(36)
        close_btn.clicked.connect(self.close)
        control_layout.addWidget(close_btn)

        main_layout.addWidget(self.control_panel)

        # Table area
        self.table = QTableWidget(main_widget)
        self.table.keyPressEvent = self._table_key_press_event(self.table)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background: {self.colors['bg_primary']};
                color: {self.colors['text_primary']};
                border: 1px solid {self.colors['border']};
                border-radius: 24px; padding: 24px 24px;
            }}
            QTableWidget::item:selected {{
                background: {self.colors['selection']};
                color: #000000;
                border: 1px solid #555555;
            }}
            QHeaderView::section {{
                background: {self.colors['bg_primary_accent']};
                color: {self.colors['text_primary']};
                border: 1px solid {self.colors['border']};
                border-radius: 24px; padding: 6px 6px;
            }}
        """)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.doubleClicked.connect(self.expand_item)

        main_layout.addWidget(self.table, stretch=1)

        # Status bar
        status_panel = QFrame(main_widget)
        status_panel.setStyleSheet(f"background: {self.colors['bg_secondary']}; border: 1px solid {self.colors['border']}; border-radius: 22px; padding: 6px 6px;")
        status_layout = QHBoxLayout(status_panel)
        status_layout.setContentsMargins(10, 2, 10, 2)

        self.status_label = QLabel("Ready", status_panel)
        self.status_label.setStyleSheet(f"background: transparent; color: {self.colors['text_secondary']}; border : none;")
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        self.row_count_label = QLabel("", status_panel)
        self.row_count_label.setStyleSheet(f"background: transparent; color: {self.colors['text_secondary']}; border : none;")
        status_layout.addWidget(self.row_count_label)

        main_layout.addWidget(status_panel)

    def delete_selected_rows(self):
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            QMessageBox.warning(self, "No selection", "Select at least one row to delete.")
            return
        # First confirmation
        if QMessageBox.question(self, "Confirm Deletion", f"Are you sure you want to delete {len(selected)} row(s)?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        # Second confirmation
        if QMessageBox.question(self, "Confirm Again", "Do you pinky-promise that you're sure ?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        # Save current state for undo
        import copy
        self.undo_stack.append(self.table_manager.df.copy(deep=True))
        self.redo_stack.clear()
        # Delete from DataFrame by DataFrame index
        indices_to_delete = []
        for idx in selected:
            item = self.table.item(idx.row(), 0)
            if item is not None:
                df_index = item.data(Qt.ItemDataRole.UserRole)
                indices_to_delete.append(df_index)
        self.table_manager.df = self.table_manager.df.drop(indices_to_delete)
        self.table_manager.df.reset_index(drop=True, inplace=True)
        self.table_manager.filtered_df = self.table_manager.df.copy()
        self.table_manager.populate_table()
        self.dirty = True
        # Removed duplicate UI/layout code that was outside init_ui (fixed scoping errors)

    def _create_line_edit(self, parent, placeholder=None):
        line = QLineEdit(parent)
        line.setStyleSheet(f"""
            QLineEdit {{
                background: {self.colors['bg_primary']};
                color: {self.colors['text_primary']};
                border: 1px solid {self.colors['border']};
                border-radius: 12px;
                padding: 6px 12px;
            }}
        """)
        if placeholder:
            line.setPlaceholderText(placeholder)
        return line

    def _create_label(self, parent, text):
        label = QLabel(text, parent)
        label.setStyleSheet(f"""
            QLabel {{
                background: transparent;
                color: {self.colors['text_secondary']};
                border: none;
            }}
        """)
        return label

    def _create_button(self, parent, text):
        btn = QPushButton(text, parent)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {self.colors['bg_primary']};
                color: {self.colors['text_primary']};
                border: 1px solid {self.colors['border']};
                border-radius: 16px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {self.colors['bg_primary_accent']};
            }}
            QPushButton:pressed {{
                background: {self.colors.get('bg_primary_pressed', self.colors['bg_primary_accent'])};
            }}
            QPushButton:checked {{
                background: {self.colors['bg_primary_accent']};
            }}
        """)
        return btn
    
    def _table_key_press_event(self, table):
        orig = table.keyPressEvent
        def handler(event):
            if event.matches(QKeySequence.StandardKey.Copy):
                copy_selected_cells(table)
            else:
                orig(event)
        return handler

    def save_file(self):
        if self.table_manager.df is not None and self.table_manager.current_file:
            try:
                normalize_operations_column(self.table_manager.df)  # <--- Always normalize before saving
                self.table_manager.df.to_parquet(self.table_manager.current_file)
                self.dirty = False
                self.status_label.setText("File saved.")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save file: {str(e)}")
        else:
            QMessageBox.warning(self, "No file", "No file loaded to save.")

    def fix_operations_column(df):
        def fix_ops(ops):
            # Handle both list and numpy array of dicts
            if isinstance(ops, (list, tuple)):
                return [
                    OrderedDict([
                        ("ordre", op.get("ordre", "")),
                        ("description", op.get("description", "")),
                        ("url", op.get("url", "")),
                    ])
                    for op in ops if isinstance(op, dict)
                ]
            elif hasattr(ops, "__iter__"):  # numpy array
                return [
                    OrderedDict([
                        ("ordre", op.get("ordre", "")),
                        ("description", op.get("description", "")),
                        ("url", op.get("url", "")),
                    ])
                    for op in list(ops) if isinstance(op, dict)
                ]
            else:
                return []
        if "operations" in df.columns:
            df["operations"] = df["operations"].apply(fix_ops)

    def closeEvent(self, event):
        if self.dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Save before exiting?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.save_file()
                event.accept()
            elif reply == QMessageBox.StandardButton.No:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'underlay_frame'):
            self.underlay_frame.setGeometry(self.rect())
            self.underlay_frame.lower()
        if hasattr(self, 'bg_frame'):
            self.bg_frame.setGeometry(self.rect())
            self.bg_frame.lower()
        if hasattr(self, 'table') and self.table.columnCount() > 0:
            QTimer.singleShot(50, lambda: smart_resize_columns(self.table))
        if self.overlay:
            self.overlay.setGeometry(self.rect())
            if self.search_widget:
                widget_width = 600
                widget_height = 500
                x = (self.width() - widget_width) // 2
                y = (self.height() - widget_height) // 2
                self.search_widget.setGeometry(x, y, widget_width, widget_height)
        if self.glass_overlay:
            self.glass_overlay.setGeometry(self.rect())
            if self.glass_widget:
                widget_width = int(self.width() * 0.8)
                widget_height = int(self.height() * 0.8)
                x = int(self.width() * 0.1)
                y = int(self.height() * 0.1)
                self.glass_widget.setGeometry(x, y, widget_width, widget_height)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Parquet File", "", "Parquet files (*.parquet);;All files (*)")
        if file_path:
            try:
                self.table_manager.load_parquet_file(file_path)
                self.settings.setValue("last_file", file_path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
    
    def toggle_max_restore(self):
        """
        Toggle between maximized and normal window state.
        Works regardless of how the window was maximized.
        """
        screen = self.screen()
        available_geometry = screen.availableGeometry()
        current_geometry = self.geometry()
        tolerance = 10
        is_effectively_maximized = (
            abs(current_geometry.x() - available_geometry.x()) <= tolerance and
            abs(current_geometry.y() - available_geometry.y()) <= tolerance and
            abs(current_geometry.width() - available_geometry.width()) <= tolerance and
            abs(current_geometry.height() - available_geometry.height()) <= tolerance
        )
        if is_effectively_maximized or self.isMaximized():
            # Restore to startup geometry
            self.showNormal()
            self.setGeometry(self._startup_geometry)
        else:
            if not self.isMaximized():
                self._normal_geometry = self.geometry()
            self.showMaximized()

    def open_search(self):
        if self.table_manager.df is None:
            QMessageBox.warning(self, "Warning", "Please load a Parquet file first")
            return
        self.create_search_overlay()

    def create_search_overlay(self):
        if self.overlay:
            self.overlay.deleteLater()
        self.overlay = QFrame(self)
        self.overlay.setStyleSheet(f"""
            QFrame {{
                {self.colors['bg_primary']};
                border: none;
            }}
        """)
        self.overlay.setGeometry(self.rect())
        self.overlay.show()
        self.search_widget = SearchWidget(self.overlay, self.table_manager.df, self.colors, filters=self.table_manager.filters)
        widget_width = 600
        widget_height = 500
        x = (self.width() - widget_width) // 2
        y = (self.height() - widget_height) // 2
        self.search_widget.setGeometry(x, y, widget_width, widget_height)
        self.search_widget.show()
        self.search_widget.filters_applied.connect(self.on_search_applied)
        self.search_widget.closed.connect(self.close_search_overlay)
        self.overlay.installEventFilter(self)
        self.overlay.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.overlay.setFocus()

    def close_search_overlay(self):
        if self.overlay:
            self.overlay.deleteLater()
            self.overlay = None
            self.search_widget = None

    def on_search_applied(self, filters):
        self.table_manager.filters = filters
        self.table_manager.apply_filters()
        self.close_search_overlay()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self.search_widget:
                self.close_search_overlay()
            elif self.glass_widget:
                self.close_glass_overlay()
        elif event.key() == Qt.Key.Key_Down and self.glass_widget:
            self.show_next_glass()
            return
        elif event.key() == Qt.Key.Key_Up and self.glass_widget:
            self.show_prev_glass()
            return
        elif event.key() == Qt.Key.Key_Right and self.glass_widget:
            self.show_next_glass()
            return
        elif event.key() == Qt.Key.Key_Left and self.glass_widget:
            self.show_prev_glass()
            return
        else:
            super().keyPressEvent(event)
    
    def expand_item(self, index):
        visual_row = index.row()
        # Map visual row to DataFrame index using UserRole
        item = self.table.item(visual_row, 0)
        if item is not None:
            df_index = item.data(Qt.ItemDataRole.UserRole)
            # Find the position of this df_index in filtered_df
            try:
                filtered_indices = list(self.table_manager.filtered_df.index)
                row_pos = filtered_indices.index(df_index)
            except Exception:
                row_pos = visual_row
        else:
            row_pos = visual_row
        self.filtered_row_indices = list(self.table_manager.filtered_df.index)
        self.current_glass_row = row_pos
        # Use the correct DataFrame row for item_data
        data = self.table_manager.filtered_df.iloc[row_pos].to_dict() if 0 <= row_pos < len(self.table_manager.filtered_df) else {}
        self.create_glass_overlay(data)
    
    def show_next_glass(self):
        # Move to the next visual row in the table (wrap if needed)
        current_visual_row = self.table.currentRow()
        if current_visual_row == -1:
            current_visual_row = 0
        next_visual_row = (current_visual_row + 1) % self.table.rowCount()
        item = self.table.item(next_visual_row, 0)
        if item is not None:
            df_index = item.data(Qt.ItemDataRole.UserRole)
            filtered_indices = list(self.table_manager.filtered_df.index)
            try:
                row_pos = filtered_indices.index(df_index)
            except Exception:
                row_pos = next_visual_row
            self.current_glass_row = row_pos
            data = self.table_manager.filtered_df.iloc[row_pos].to_dict()
            self.create_glass_overlay(data)
            self.select_table_row(row_pos)
    
    def show_prev_glass(self):
        # Move to the previous visual row in the table (wrap if needed)
        current_visual_row = self.table.currentRow()
        if current_visual_row == -1:
            current_visual_row = 0
        prev_visual_row = (current_visual_row - 1) % self.table.rowCount()
        item = self.table.item(prev_visual_row, 0)
        if item is not None:
            df_index = item.data(Qt.ItemDataRole.UserRole)
            filtered_indices = list(self.table_manager.filtered_df.index)
            try:
                row_pos = filtered_indices.index(df_index)
            except Exception:
                row_pos = prev_visual_row
            self.current_glass_row = row_pos
            data = self.table_manager.filtered_df.iloc[row_pos].to_dict()
            self.create_glass_overlay(data)
            self.select_table_row(row_pos)

    def select_table_row(self, row):
        """Selects and scrolls to the given row in the main table, accounting for sorting."""
        # Find the DataFrame index for the row in filtered_df
        if 0 <= row < len(self.table_manager.filtered_df):
            df_index = self.table_manager.filtered_df.index[row]
            # Find the visual row in the table that matches this df_index
            for visual_row in range(self.table.rowCount()):
                item = self.table.item(visual_row, 0)
                if item is not None and item.data(Qt.ItemDataRole.UserRole) == df_index:
                    self.table_manager.table.setCurrentCell(visual_row, 0)
                    self.table_manager.table.scrollToItem(
                        self.table_manager.table.item(visual_row, 0),
                        QAbstractItemView.ScrollHint.PositionAtCenter
                    )
                    break

    def create_glass_overlay(self, data):
        if self.glass_overlay:
            self.glass_overlay.deleteLater()
        self.glass_overlay = QFrame(self)
        self.glass_overlay.setStyleSheet(f"""
            QFrame {{
                {self.colors['bg_secondary']};
                border: none;
            }}
        """)
        self.glass_overlay.setGeometry(self.rect())
        self.glass_overlay.show()
        self.glass_widget = GlassWidget(self.glass_overlay, data, self.colors)
        widget_width = int(self.width() * 0.8)
        widget_height = int(self.height() * 0.8)
        x = int(self.width() * 0.1)
        y = int(self.height() * 0.1)
        self.glass_widget.setGeometry(x, y, widget_width, widget_height)
        self.glass_widget.show()
        self.glass_widget.closed.connect(self.close_glass_overlay)
        self.glass_overlay.installEventFilter(self)
        self.glass_overlay.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.glass_overlay.setFocus()

    def close_glass_overlay(self):
        if self.glass_overlay:
            self.glass_overlay.deleteLater()
            self.glass_overlay = None
            self.glass_widget = None

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._dragging:
            diff = event.globalPosition().toPoint() - self._drag_start_pos
            new_pos = self._drag_start_geom.topLeft() + diff
            self.move(new_pos)
        elif self._resizing and self._resize_dir:
            self._resize_window(event.globalPosition().toPoint())
        else:
            # Only set 4-headed arrow if in drag area (control_panel)
            if self._is_in_drag_area(pos):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                resize_dir = self._get_resize_direction(pos)
                if resize_dir:
                    self.setCursor(self._cursor_for_direction(resize_dir))
                else:
                    self.unsetCursor()  # Let widgets show their own cursor
        super().mouseMoveEvent(event)
        if self._dragging:
            diff = event.globalPosition().toPoint() - self._drag_start_pos
            new_pos = self._drag_start_geom.topLeft() + diff
            self.move(new_pos)
        elif self._resizing and self._resize_dir:
            self._resize_window(event.globalPosition().toPoint())
        else:
            # Only set 4-headed arrow if in drag area (control_panel)
            if self._is_in_drag_area(event.position().toPoint()):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                resize_dir = self._get_resize_direction(event.position().toPoint())
                if resize_dir:
                    self.setCursor(self._cursor_for_direction(resize_dir))
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._is_in_drag_area(event.position().toPoint()):
                self._dragging = True
                self._drag_start_pos = event.globalPosition().toPoint()
                self._drag_start_geom = self.geometry()
                return
            self._resize_dir = self._get_resize_direction(event.position().toPoint())
            if self._resize_dir:
                self._resizing = True
                self._mouse_press_pos = event.globalPosition().toPoint()
                self._mouse_press_geom = self.geometry()
            else:
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._resizing = False
        self._dragging = False
        self._resize_dir = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        cursor_pos = QCursor.pos()
        screen = QApplication.primaryScreen().availableGeometry()
        margin = 20
        x, y = cursor_pos.x(), cursor_pos.y()
        w, h = screen.width(), screen.height()
        if abs(x - screen.left()) < margin and abs(y - screen.top()) < margin:
            self.setGeometry(screen.left(), screen.top(), w // 2, h // 2)
        elif abs(x - screen.right()) < margin and abs(y - screen.top()) < margin:
            self.setGeometry(screen.left() + w // 2, screen.top(), w // 2, h // 2)
        elif abs(x - screen.left()) < margin and abs(y - screen.bottom()) < margin:
            self.setGeometry(screen.left(), screen.top() + h // 2, w // 2, h // 2)
        elif abs(x - screen.right()) < margin and abs(y - screen.bottom()) < margin:
            self.setGeometry(screen.left() + w // 2, screen.top() + h // 2, w // 2, h // 2)
        elif abs(x - screen.left()) < margin:
            self.setGeometry(screen.left(), screen.top(), w // 2, h)
        elif abs(x - screen.right()) < margin:
            self.setGeometry(screen.left() + w // 2, screen.top(), w // 2, h)
        elif abs(y - screen.top()) < margin:
            self.setGeometry(screen)
        super().mouseReleaseEvent(event)

    def _get_resize_direction(self, pos):
        x, y, w, h = pos.x(), pos.y(), self.width(), self.height()
        margin = self.BORDER_WIDTH
        directions = []
        if x < margin:
            directions.append('left')
        elif x > w - margin:
            directions.append('right')
        if y < margin:
            directions.append('top')
        elif y > h - margin:
            directions.append('bottom')
        return '-'.join(directions) if directions else None

    def _cursor_for_direction(self, direction):
        cursors = {
            'left': Qt.CursorShape.SizeHorCursor,
            'right': Qt.CursorShape.SizeHorCursor,
            'top': Qt.CursorShape.SizeVerCursor,
            'bottom': Qt.CursorShape.SizeVerCursor,
            'left-top': Qt.CursorShape.SizeFDiagCursor,
            'top-left': Qt.CursorShape.SizeFDiagCursor,
            'right-bottom': Qt.CursorShape.SizeFDiagCursor,
            'bottom-right': Qt.CursorShape.SizeFDiagCursor,
            'right-top': Qt.CursorShape.SizeBDiagCursor,
            'top-right': Qt.CursorShape.SizeBDiagCursor,
            'left-bottom': Qt.CursorShape.SizeBDiagCursor,
            'bottom-left': Qt.CursorShape.SizeBDiagCursor,
        }
        return cursors.get(direction, Qt.CursorShape.ArrowCursor)

    def _resize_window(self, global_pos):
        diff = global_pos - self._mouse_press_pos
        geom = self._mouse_press_geom
        min_width, min_height = 600, 400
        x, y, w, h = geom.x(), geom.y(), geom.width(), geom.height()
        dx, dy = diff.x(), diff.y()
        dir = self._resize_dir
        if 'right' in dir:
            w = max(min_width, w + dx)
        if 'bottom' in dir:
            h = max(min_height, h + dy)
        if 'left' in dir:
            x = x + dx
            w = max(min_width, w - dx)
        if 'top' in dir:
            y = y + dy
            h = max(min_height, h - dy)
        self.setGeometry(x, y, w, h)

    def _is_in_drag_area(self, pos):
        if not hasattr(self, "control_panel"):
            return False
        mapped = self.control_panel.mapFrom(self, pos)
        return self.control_panel.rect().contains(mapped)

    def eventFilter(self, obj, event):
        if obj == self.centralWidget():
            if event.type() == QEvent.Type.MouseMove:
                self.mouseMoveEvent(event)
                return True
            elif event.type() == QEvent.Type.MouseButtonPress:
                self.mousePressEvent(event)
                return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                self.mouseReleaseEvent(event)
                return True
        if obj == self.overlay and event.type() == QEvent.Type.MouseButtonPress:
            if self.search_widget and not self.search_widget.geometry().contains(event.position().toPoint()):
                self.close_search_overlay()
                return True
        if obj == self.glass_overlay and event.type() == QEvent.Type.MouseButtonPress:
            if self.glass_widget and not self.glass_widget.geometry().contains(event.position().toPoint()):
                self.close_glass_overlay()
                return True
        if hasattr(self, 'editor_overlay') and obj == self.editor_overlay and event.type() == QEvent.Type.MouseButtonPress:
            if self.operations_editor and not self.operations_editor.geometry().contains(event.position().toPoint()):
                self.operations_editor.closed.emit(False)
                return True
        if hasattr(self, 'addrow_overlay') and obj == self.addrow_overlay and event.type() == QEvent.Type.MouseButtonPress:
            if not self.addrow_widget.geometry().contains(event.position().toPoint()):
                self.addrow_overlay.hide()
                self.addrow_overlay.deleteLater()
                return True
        return super().eventFilter(obj, event)

    def edit_selected_rows(self):
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            QMessageBox.warning(self, "No selection", "Select at least one row to edit.")
            return
        rows = [idx.row() for idx in selected]
        items = [self.table_manager.item_data[r] for r in rows]
        self.open_operations_editor(items, rows)  # <-- pass rows

    def open_operations_editor(self, items, rows):
        # Create overlay
        self.editor_overlay = QFrame(self)
        self.editor_overlay.setGeometry(self.rect())
        self.editor_overlay.show()
        # Create editor widget as child of overlay
        self.operations_editor = operationsListEditor(self.editor_overlay, items, self.colors)
        # Center the editor
        editor_width = int(self.width() * 0.9)
        editor_height = int(self.height() * 0.9)
        x = (self.width() - editor_width) // 2
        y = (self.height() - editor_height) // 2
        self.operations_editor.setGeometry(x, y, editor_width, editor_height)
        self.operations_editor.show()
        self.operations_editor.setFocus()
        # Install event filter for outside clicks
        self.editor_overlay.installEventFilter(self)
        # Connect close signal, now passing rows
        self.operations_editor.closed.connect(
            lambda accepted: self.handle_editor_closed(self.operations_editor, rows, accepted)
        )
    
    def get_updated_lists(self):
        """
        Returns a list of lists, each containing OrderedDicts for the operations of each edited row.
        The order matches the order of the items/rows passed to the editor.
        """
        results = []
        if self.edit_all and self.all_table and self.all_table.isVisible():
            table = self.all_table
            ops = []
            for r in range(table.rowCount()):
                desc_item = table.item(r, 1)
                url_item = table.item(r, 2)
                desc = desc_item.text() if desc_item else ""
                if desc == "#N/A":
                    desc = ""
                url = url_item.text() if url_item else ""
                if url == "#N/A":
                    url = ""
                if desc or url:
                    ops.append(OrderedDict([
                        ("ordre", r + 1),
                        ("description", desc),
                        ("url", url)
                    ]))
            results = [copy.deepcopy(ops) for _ in self.items]
        elif hasattr(self, "tables") and self.tables:
            for i, table in enumerate(self.tables):
                ops = []
                for r in range(table.rowCount()):
                    desc_item = table.item(r, 1)
                    url_item = table.item(r, 2)
                    desc = desc_item.text() if desc_item else ""
                    if desc == "#N/A":
                        desc = ""
                    url = url_item.text() if url_item else ""
                    if url == "#N/A":
                        url = ""
                    if desc or url:
                        ops.append(OrderedDict([
                            ("ordre", r + 1),
                            ("description", desc),
                            ("url", url)
                        ]))
                results.append(ops)
        return results

    def handle_editor_closed(self, dlg, rows, accepted):
        if hasattr(self, 'editor_overlay'):
            self.editor_overlay.deleteLater()
            del self.editor_overlay
        if hasattr(self, 'operations_editor'):
            del self.operations_editor
        if accepted:
            # --- UNDO/REDO SUPPORT ---
            import copy
            self.undo_stack.append(copy.deepcopy(self.table_manager.df))
            self.redo_stack.clear()
            # --- END UNDO/REDO SUPPORT ---
            updated_lists = dlg.get_updated_lists()
            if getattr(dlg, "edit_all", False) and len(updated_lists) > 0:
                batch_ops = copy.deepcopy(updated_lists[0])
                for row_idx in rows:
                    filtered_index = self.table_manager.filtered_df.index[row_idx]
                    self.table_manager.df.at[filtered_index, 'operations'] = batch_ops
            else:
                for i, row_idx in enumerate(rows):
                    filtered_index = self.table_manager.filtered_df.index[row_idx]
                    self.table_manager.df.at[filtered_index, 'operations'] = [
                        OrderedDict([
                            ("ordre", op.get('ordre', '')),
                            ("description", op.get('description', '')),
                            ("url", op.get('url', ''))
                        ])
                        for op in updated_lists[i]
                    ]
                # Update the date field to now
                now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
                self.table_manager.df.at[filtered_index, 'date'] = now
            # Normalize after all updates
            normalize_operations_column(self.table_manager.df)
            self.table_manager.apply_filters()
            self.dirty = True
        if hasattr(self, 'editor_overlay'):
            self.editor_overlay.deleteLater()
            self.editor_overlay = None

    def undo(self):
        if not self.undo_stack:
            return
        import copy
        self.redo_stack.append(copy.deepcopy(self.table_manager.df))
        self.table_manager.df = self.undo_stack.pop()
        self.table_manager.filtered_df = self.table_manager.df.copy()
        self.table_manager.populate_table()
        self.dirty = True
        self.status_label.setText("Undo")

    def redo(self):
        if not self.redo_stack:
            return
        import copy
        self.undo_stack.append(copy.deepcopy(self.table_manager.df))
        self.table_manager.df = self.redo_stack.pop()
        self.table_manager.filtered_df = self.table_manager.df.copy()
        self.table_manager.populate_table()
        self.dirty = True
        self.status_label.setText("Redo")
    
    def fix_operations_column(df):
        def fix_ops(ops):
            # Handle both list and numpy array of dicts
            if isinstance(ops, (list, tuple)):
                return [
                    OrderedDict([
                        ("ordre", op.get("ordre", "")),
                        ("description", op.get("description", "")),
                        ("url", op.get("url", "")),
                    ])
                    for op in ops if isinstance(op, dict)
                ]
            elif hasattr(ops, "__iter__"):  # numpy array
                return [
                    OrderedDict([
                        ("ordre", op.get("ordre", "")),
                        ("description", op.get("description", "")),
                        ("url", op.get("url", "")),
                    ])
                    for op in list(ops) if isinstance(op, dict)
                ]
            else:
                return []
        if "operations" in df.columns:
            df["operations"] = df["operations"].apply(fix_ops)

    def create_new_row(self):
        columns = list(self.table_manager.df.columns) if self.table_manager.df is not None else []
        if not columns:
            return
        self.undo_stack.append(self.table_manager.df.copy(deep=True))
        self.redo_stack.clear()
    
        input_columns = [col for col in columns if col not in ("date", "operations")]
    
        # Create overlay
        self.addrow_overlay = QFrame(self)
        self.addrow_overlay.setGeometry(self.rect())
        self.addrow_overlay.setStyleSheet("background: rgba(0,0,0,80); border: none;")
        self.addrow_overlay.show()
    
        # Create AddRowWidget as child of overlay
        self.addrow_widget = AddRowWidget(self.addrow_overlay, input_columns, self.colors)
        self.addrow_widget.center_in_parent()  # This will center in overlay
    
        def on_accept(values):
            # ...existing code...
            self.addrow_overlay.hide()
            self.addrow_overlay.deleteLater()
    
        def on_cancel():
            self.addrow_overlay.hide()
            self.addrow_overlay.deleteLater()
    
        self.addrow_widget.accepted.connect(on_accept)
        self.addrow_widget.cancelled.connect(on_cancel)
    
        # Install event filter to close on outside click
        self.addrow_overlay.installEventFilter(self)

class AddRowWidget(QFrame):
    accepted = pyqtSignal(dict)
    cancelled = pyqtSignal()
    closed = pyqtSignal()

    def __init__(self, parent, columns, colors):
        super().__init__(parent)
        self.colors = colors
        self.columns = columns
        self.widgets = {}
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        
        # Make it a proper overlay widget
        self.setParent(parent)
        self.setWindowFlags(Qt.WindowType.Widget)
        
        self.setStyleSheet(f"""
            QFrame {{
                background: {colors['bg_secondary']};
                border: 1px solid {colors['border']};
                border-radius: 32px;
            }}
        """)

        # Layout setup
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(32, 32, 32, 32)
        main_layout.setSpacing(8)

        # Title (no border)
        title = QLabel("Add New Row", self)
        title.setStyleSheet(f"color: {colors['text_primary']}; background: transparent; font-weight: bold; font-size: 22px; border: none;")
        main_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignLeft)

        # Input fields with labels directly on top
        for col in columns:
            field_layout = QVBoxLayout()
            field_layout.setSpacing(0)  # No extra space between label and input
            label = QLabel(col, self)
            label.setStyleSheet(f"color: {colors['text_primary']}; background: transparent; font-size: 15px; border: none; margin-bottom: 0px;")
            field_layout.addWidget(label)
            line_edit = QLineEdit(self)
            line_edit.setStyleSheet(f"""
                QLineEdit {{
                    background: {colors['bg_secondary']};
                    color: {colors['text_primary']};
                    border: 1px solid {colors['border']};
                    padding: 2px 8px;
                }}
            """)
            self.widgets[col] = line_edit
            field_layout.addWidget(line_edit)
            main_layout.addLayout(field_layout)

        # Buttons
        btns_layout = QHBoxLayout()
        btns_layout.addStretch()
        
        ok_btn = HoverGlowButton("OK", self, self.colors)
        cancel_btn = HoverGlowButton("Cancel", self, self.colors)
        ok_btn.setFixedWidth(120)
        cancel_btn.setFixedWidth(120)
        
        btns_layout.addWidget(ok_btn)
        btns_layout.addWidget(cancel_btn)
        main_layout.addLayout(btns_layout)

        ok_btn.clicked.connect(self._on_ok)
        cancel_btn.clicked.connect(self._on_cancel)

        # Center and show
        self.center_in_parent()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()
        self.raise_()
        self.show()

    def center_in_parent(self):
        if not self.parent():
            return
        parent = self.parent()
        parent_rect = parent.rect()
        widget_width = min(500, int(parent_rect.width() * 0.75))
        widget_height = min(400, int(parent_rect.height() * 0.75))
        self.setFixedSize(widget_width, widget_height)
        x = (parent_rect.width() - widget_width) // 2
        y = (parent_rect.height() - widget_height) // 2
        self.move(x, y)

    def _on_ok(self):
        values = {}
        for col, widget in self.widgets.items():
            val = widget.text().strip()
            if col == "num_seq_op":
                if not val:
                    self._show_message("Missing Field", "num_seq_op is required.")
                    return
                try:
                    int_val = int(val)
                except Exception:
                    self._show_message("Invalid Input", "num_seq_op must be an integer.")
                    return
                values[col] = int_val
            elif col == "code_phase":
                if not val:
                    self._show_message("Missing Field", "code_phase is required.")
                    return
                values[col] = val
            else:
                values[col] = val
        self.accepted.emit(values)
        self.close_widget()

    def _on_cancel(self):
        self.cancelled.emit()
        self.close_widget()

    def close_widget(self):
        self.closed.emit()
        self.close()

    def _show_message(self, title, text):
        QMessageBox.warning(self, title, text)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close_widget()
            event.accept()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        # This handles clicks on the widget itself
        super().mousePressEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        # Install event filter on parent to catch outside clicks
        if self.parent():
            self.parent().installEventFilter(self)

    def hideEvent(self, event):
        super().hideEvent(event)
        # Remove event filter when hiding
        if self.parent():
            self.parent().removeEventFilter(self)

    def eventFilter(self, obj, event):
        # Handle clicks on the parent (outside this widget)
        if obj == self.parent() and event.type() == event.Type.MouseButtonPress:
            # Check if click is outside this widget
            click_pos = event.pos()
            widget_rect = self.geometry()
            
            if not widget_rect.contains(click_pos):
                self.close_widget()
                return True
                
        return super().eventFilter(obj, event)

class SearchWidget(QFrame):
    filters_applied = pyqtSignal(dict)
    closed = pyqtSignal()
    
    def __init__(self, parent, df, colors, filters=None):
        super().__init__(parent)
        self.colors = COLORS
        self.df = df
        self.filters = dict(filters) if filters else {}
        
        self.setStyleSheet(f"""
            QFrame {{
                background: {colors['bg_secondary']};
                border: 1px solid {colors['border']};
                border-radius: 32px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        # Title with close button
        title_frame = QHBoxLayout()
        title = QLabel("Advanced Search & Filter", self)
        title.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {colors['text_primary']}; border: none;")
        title_frame.addWidget(title)
        
        layout.addLayout(title_frame)

        # Rest of the layout (same as SearchDialog but without modal setup)
        filter_frame = QFrame(self)
        filter_frame.setStyleSheet(f"background: {colors['bg_secondary_accent']}; border: 1px solid {colors['border']}; border-radius: 15px;")
        filter_layout = QVBoxLayout(filter_frame)
        filter_layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(filter_frame)

        col_frame = QHBoxLayout()
        filter_layout.addLayout(col_frame)
        col_label = QLabel("Column:", self)
        col_label.setStyleSheet(f"color: {colors['text_primary']}; border: none;")
        col_frame.addWidget(col_label)
        self.column_combo = QComboBox(self)
        self.column_combo.addItems(list(df.columns))
        self.column_combo.setStyleSheet(f"color: {colors['text_primary']}; background: {self.colors['bg_secondary']}; border: 1px solid {colors['border']};")
        col_frame.addWidget(self.column_combo)

        op_frame = QHBoxLayout()
        filter_layout.addLayout(op_frame)
        op_label = QLabel("Operation:", self)
        op_label.setStyleSheet(f"color: {colors['text_primary']}; border: none;")
        op_frame.addWidget(op_label)
        self.operation_combo = QComboBox(self)
        self.operation_combo.addItems([
            "contains", "!contains", "equals", "!equals", "empty", "!empty", 
            "starts_with", "ends_with", "greater_than", "less_than", 
            "date_before", "date_after", "date_between"
        ])
        self.operation_combo.setStyleSheet(f"color: {colors['text_primary']}; background: {self.colors['bg_secondary']}; border: 1px solid {colors['border']};")
        op_frame.addWidget(self.operation_combo)
        
        value_frame = QHBoxLayout()
        filter_layout.addLayout(value_frame)
        value_label = QLabel("Value:", self)
        value_label.setStyleSheet(f"color: {colors['text_primary']}; border: none;")
        value_frame.addWidget(value_label)
        self.value_edit = QLineEdit(self)
        self.operation_combo.currentTextChanged.connect(self.update_value_field_state) #Needs to be placed after qlinedit and before updatevaluefieldstate
        self.value_edit.setStyleSheet(f"color: {colors['text_primary']}; background: {self.colors['bg_secondary']}; border: 1px solid {colors['border']};")
        self.value_edit.returnPressed.connect(self.add_filter)  # Allow Enter to add filter
        value_frame.addWidget(self.value_edit)

        # Case sensitivity toggle
        case_frame = QHBoxLayout()
        filter_layout.addLayout(case_frame)
        self.case_sensitive_checkbox = QCheckBox("Case Sensitive", self)
        self.case_sensitive_checkbox.setChecked(False)  # Default to case-insensitive
        self.case_sensitive_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {colors['text_primary']};
                border: none;
                spacing: 5px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {colors['border']};
                border-radius: 5px;
                background: {colors['bg_secondary']};
            }}
            QCheckBox::indicator:checked {{
                background: {colors['bg_primary']};
                border: 1px solid {colors['border']};
            }}
            QCheckBox::indicator:checked:hover {{
                background: {colors['border']};
            }}
        """)
        case_frame.addWidget(self.case_sensitive_checkbox)
        case_frame.addStretch()  # Push checkbox to the left

        btn_frame = QHBoxLayout()
        filter_layout.addLayout(btn_frame)

        add_btn = HoverGlowButton("Add", self, self.colors)
        add_btn.clicked.connect(self.add_filter)
        btn_frame.addWidget(add_btn)

        clear_btn = HoverGlowButton("Clear", self, self.colors)
        clear_btn.clicked.connect(self.clear_filters)
        btn_frame.addWidget(clear_btn)

        apply_btn = HoverGlowButton("Apply", self, self.colors)
        apply_btn.clicked.connect(self.apply_filters)
        btn_frame.addWidget(apply_btn)


        self.filters_list = QListWidget(self)
        self.filters_list.setStyleSheet(f"""
            background: {self.colors['bg_secondary_accent']};
            color: {colors['text_secondary']};
            border: 1px solid {colors['border']};
            border-radius: 22px;
            padding: 10px;
        """)
        self.filters_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.filters_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.filters_list.customContextMenuRequested.connect(self.show_filter_context_menu)
        layout.addWidget(self.filters_list, stretch=1)

        self.update_filters_display()
        self.update_value_field_state()

    def add_filter(self):
        column = self.column_combo.currentText()
        operation = self.operation_combo.currentText()
        value = self.value_edit.text()
        case_sensitive = self.case_sensitive_checkbox.isChecked()
        if not column:
            return
        operation = self.operation_combo.currentText()
        value_required = operation not in ["empty", "!empty"]
        if value_required and not self.value_edit.text().strip():
            return
        filter_key = f"{column}_{operation}_{len(self.filters)}"
        self.filters[filter_key] = {
            'column': column,
            'operation': operation,
            'value': value,
            'case_sensitive': case_sensitive
        }
        self.update_filters_display()
        self.value_edit.clear()

    def clear_filters(self):
        self.filters.clear()
        self.update_filters_display()

    def update_filters_display(self):
        self.filters_list.clear()
        if not self.filters:
            self.filters_list.addItem("No active filters")
            self.filters_list.setEnabled(False)
            return
        self.filters_list.setEnabled(True)
        string_ops = ["contains", "equals", "starts_with", "ends_with"]
        for key, filter_data in self.filters.items():
            case_info = " (case-sensitive)" if filter_data.get('case_sensitive', True) else " (case-insensitive)"
            if filter_data['operation'] in string_ops:
                text = f"{filter_data['column']} {filter_data['operation']} '{filter_data['value']}'{case_info}"
            else:
                text = f"{filter_data['column']} {filter_data['operation']} '{filter_data['value']}'"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, key)  # Store filter key for removal
            self.filters_list.addItem(item)

    def show_filter_context_menu(self, pos):
        item = self.filters_list.itemAt(pos)
        if item is None or not self.filters_list.isEnabled():
            return

        menu = QMenu(self)
        delete_action = menu.addAction("Delete Filter")
        action = menu.exec(self.filters_list.mapToGlobal(pos))
        if action == delete_action:
            self.remove_selected_filter()

    def apply_filters(self):
        # checks for empty va&lues on filters that need it
        column = self.column_combo.currentText()
        operation = self.operation_combo.currentText()
        value = self.value_edit.text()
        case_sensitive = self.case_sensitive_checkbox.isChecked()
        
        value_required = operation not in ["empty", "!empy"]
        if column and (not value_required or value):
            filter_key = f"{column}_{operation}_{len(self.filters)}"
            self.filters[filter_key] = {
                'column': column,
                'operation': operation,
                'value': value,
                'case_sensitive': case_sensitive
            }
        self.filters_applied.emit(self.filters)

    def remove_selected_filter(self):
        selected_items = self.filters_list.selectedItems()
        if not selected_items:
            return
        item = selected_items[0]
        key = item.data(Qt.ItemDataRole.UserRole)
        if key in self.filters:
            del self.filters[key]
        self.update_filters_display()

    def update_value_field_state(self):
        operation = self.operation_combo.currentText()
        if operation in ["empty", "!empty"]:
            self.value_edit.clear()
            self.value_edit.setDisabled(True)
            self.value_edit.setStyleSheet(f"""
                color: {self.colors['text_secondary']};
                background: {self.colors['bg_secondary_accent']};
                border: 1px solid {self.colors['border']};
            """)
        else:
            self.value_edit.setDisabled(False)
            self.value_edit.setStyleSheet(f"""
                color: {self.colors['text_primary']};
                background: {self.colors['bg_secondary']};
                border: 1px solid {self.colors['border']};
            """)

    def close_widget(self):
        self.closed.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close_widget()
        elif event.key() == Qt.Key.Key_Delete:
            # Delete selected filter if list is focused
            if self.filters_list.hasFocus():
                self.remove_selected_filter()
        else:
            super().keyPressEvent(event)

    def show_next_glass(self):
        if self.current_glass_row is None or not self.filtered_row_indices:
            return
        idx = self.filtered_row_indices.index(self.table_manager.filtered_df.index[self.current_glass_row])
        next_idx = (idx + 1) % len(self.filtered_row_indices)
        next_row = self.filtered_row_indices[next_idx]
        self.current_glass_row = self.table_manager.filtered_df.index.get_loc(next_row)
        data = self.table_manager.item_data.get(self.current_glass_row, {})
        self.create_glass_overlay(data)

    def show_prev_glass(self):
        if self.current_glass_row is None or not self.filtered_row_indices:
            return
        idx = self.filtered_row_indices.index(self.table_manager.filtered_df.index[self.current_glass_row])
        prev_idx = (idx - 1) % len(self.filtered_row_indices)
        prev_row = self.filtered_row_indices[prev_idx]
        self.current_glass_row = self.table_manager.filtered_df.index.get_loc(prev_row)
        data = self.table_manager.item_data.get(self.current_glass_row, {})
        self.create_glass_overlay(data)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ParquetViewer()
    viewer.show()
    sys.exit(app.exec())