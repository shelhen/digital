# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------
@File: main.py
@Project: latex_editor 
@Time: 2025/04/03  08:26
@Author: xieheng
@Email: xieheng@163.com
@Software: PyCharm
--------------------------------------------------------
# @Brief: 插入一段描述。
"""
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QSpinBox, QFileDialog, QGroupBox, QFormLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from io import BytesIO
import os

class LaTeXToSVGApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LaTeX 转 SVG 工具")
        self.setMinimumWidth(600)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # 输入框
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("请输入 LaTeX 公式，例如：E=mc^2")
        layout.addWidget(self.input_edit)

        # 控制面板
        control_group = QGroupBox("渲染设置")
        form_layout = QFormLayout()

        self.fontsize_spin = QSpinBox()
        self.fontsize_spin.setRange(8, 72)
        self.fontsize_spin.setValue(24)

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(300)

        form_layout.addRow("字体大小：", self.fontsize_spin)
        form_layout.addRow("DPI：", self.dpi_spin)
        control_group.setLayout(form_layout)
        layout.addWidget(control_group)

        # 按钮
        button_layout = QHBoxLayout()
        self.render_button = QPushButton("渲染公式")
        self.save_button = QPushButton("导出 SVG")
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.render_button)
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)

        # 图片显示
        self.image_label = QLabel("预览区域")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        # 信号连接
        self.render_button.clicked.connect(self.render_formula)
        self.save_button.clicked.connect(self.save_svg)

        # 存储 SVG 内容
        self.svg_data = None

    def render_formula(self):
        formula = self.input_edit.toPlainText().strip()
        fontsize = self.fontsize_spin.value()
        dpi = self.dpi_spin.value()

        if not formula:
            self.image_label.setText("请输入公式后点击渲染")
            return

        fig = plt.figure()
        fig.text(0, 0, f"${formula}$", fontsize=fontsize)

        # 自动裁剪 + 保存 SVG
        output = BytesIO()
        fig.savefig(output, format="svg", bbox_inches='tight', transparent=True, dpi=dpi)
        plt.close(fig)
        self.svg_data = output.getvalue().decode("utf-8")

        # 同时生成 PNG 供预览（临时文件）
        preview_path = "_preview.png"
        fig = plt.figure()
        fig.text(0, 0, f"${formula}$", fontsize=fontsize)
        fig.savefig(preview_path, format="png", bbox_inches='tight', transparent=True, dpi=dpi)
        plt.close(fig)

        self.image_label.setPixmap(QPixmap(preview_path))
        self.save_button.setEnabled(True)

    def save_svg(self):
        if not self.svg_data:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存 SVG", "formula.svg", "SVG Files (*.svg)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.svg_data)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LaTeXToSVGApp()
    window.show()
    sys.exit(app.exec_())