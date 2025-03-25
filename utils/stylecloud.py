# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------
@File: stylecloud_shelhen.py
@Project: digital
@Time: 2025/03/19  14:11
@Author: xieheng
@Email: xieheng@163.com
@Software: PyCharm
--------------------------------------------------------
@Brief: 插入一段描述。
"""
from __future__ import absolute_import, unicode_literals
import os
import re
import numpy as np
from matplotlib.colors import to_rgb
from wordcloud import WordCloud, ImageColorGenerator
from collections import OrderedDict
import tinycss
from shutil import rmtree
from typing import List, Union
from PIL import Image, ImageFont, ImageDraw
from six import unichr


class IconFont(object):
    """Base class that represents web icon font"""
    def __init__(self, css_file, ttf_file, keep_prefix=False):
        """
        :param css_file: path to icon font CSS file
        :param ttf_file: path to icon font TTF file
        :param keep_prefix: whether to keep common icon prefix
        """
        self.css_file = css_file
        self.ttf_file = ttf_file
        self.keep_prefix = keep_prefix
        self.css_icons, self.common_prefix = self.load_css()

    def load_css(self):
        """
        Creates a dict of all icons available in CSS file, and finds out
        what's their common prefix.
        returns sorted icons dict, common icon prefix
        """
        icons = dict()
        common_prefix = None
        parser = tinycss.make_parser('page3')
        stylesheet = parser.parse_stylesheet_file(self.css_file)

        is_icon = re.compile(r"\.(.*):before,?")

        for rule in stylesheet.rules:
            selector = rule.selector.as_css()

            # Skip CSS classes that are not icons
            if not is_icon.match(selector):
                continue

            # Find out what the common prefix is
            if common_prefix is None:
                common_prefix = selector[1:]
            else:
                common_prefix = os.path.commonprefix((common_prefix,
                                                      selector[1:]))

            for match in is_icon.finditer(selector):
                name = match.groups()[0]
                for declaration in rule.declarations:
                    if declaration.name == "content":
                        val = declaration.value.as_css()
                        # Strip quotation marks
                        if re.match(r"^['\"].*['\"]$", val):
                            val = val[1:-1]
                        icons[name] = unichr(int(val[1:], 16))

        common_prefix = common_prefix or ''

        # Remove common prefix
        if not self.keep_prefix and len(common_prefix) > 0:
            non_prefixed_icons = {}
            for name in icons.keys():
                non_prefixed_icons[name[len(common_prefix):]] = icons[name]
            icons = non_prefixed_icons

        sorted_icons = OrderedDict(sorted(icons.items(), key=lambda t: t[0]))

        return sorted_icons, common_prefix

    def export_icon(self, icon, size, color='black', scale='auto',
                    filename=None, export_dir='exported'):
        """
        Exports given icon with provided parameters.

        If the desired icon size is less than 150x150 pixels, we will first
        create a 150x150 pixels image and then scale it down, so that
        it's much less likely that the edges of the icon end up cropped.

        :param icon: valid icon name
        :param filename: name of the output file
        :param size: icon size in pixels
        :param color: color name or hex value
        :param scale: scaling factor between 0 and 1,
                      or 'auto' for automatic scaling
        :param export_dir: path to export directory
        """
        org_size = size
        size = max(150, size)

        image = Image.new("RGBA", (size, size), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        if scale == 'auto':
            scale_factor = 1
        else:
            scale_factor = float(scale)

        font = ImageFont.truetype(self.ttf_file, int(size * scale_factor))
        left, top, right, bottom = draw.textbbox((0, 0), self.css_icons[icon], font)
        width, height = right - left, bottom - top

        # If auto-scaling is enabled, we need to make sure the resulting
        # graphic fits inside the boundary. The values are rounded and may be
        # off by a pixel or two, so we may need to do a few iterations.
        # The use of a decrementing multiplication factor protects us from
        # getting into an infinite loop.
        if scale == 'auto':
            iteration = 0
            factor = 1

            while True:
                left, top, right, bottom = draw.textbbox((0, 0), self.css_icons[icon], font)
                width, height = right - left, bottom - top

                # Check if the image fits
                dim = max(width, height)
                if dim > size:
                    font = ImageFont.truetype(self.ttf_file,
                                              int(size * size/dim * factor))
                else:
                    break

                # Adjust the factor every two iterations
                iteration += 1
                if iteration % 2 == 0:
                    factor *= 0.99

        draw.text((float(size - width) / 2, float(size - height) / 2),
                  self.css_icons[icon], font=font, fill=color)

        # Get bounding box
        bbox = image.getbbox()

        # Create an alpha mask
        image_mask = Image.new("L", (size, size), 0)
        draw_mask = ImageDraw.Draw(image_mask)

        # Draw the icon on the mask
        draw_mask.text((float(size - width) / 2, float(size - height) / 2),
                       self.css_icons[icon], font=font, fill=255)

        # Create a solid color image and apply the mask
        icon_image = Image.new("RGBA", (size, size), color)
        icon_image.putalpha(image_mask)

        if bbox:
            icon_image = icon_image.crop(bbox)

        border_w = int((size - (bbox[2] - bbox[0])) / 2)
        border_h = int((size - (bbox[3] - bbox[1])) / 2)

        # Create output image
        out_image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        out_image.paste(icon_image, (border_w, border_h))

        # If necessary, scale the image to the target size
        if org_size != size:
            out_image = out_image.resize((org_size, org_size), Image.ANTIALIAS)

        # Make sure export directory exists
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        # Default filename
        if not filename:
            filename = icon + '.png'

        # Save file
        out_image.save(os.path.join(export_dir, filename))

    def get_bbox(self):
        """ 计算并获取字体文件的边界："""
        pass


def gen_palette(palette: str):
    """ 从`palettable`生成相应的调色板函数` """
    palette_split = palette.split(".")
    palette_name = palette_split[-1]
    palette_func = getattr(
        __import__(
            "palettable.{}".format(".".join(palette_split[:-1])),
            fromlist=[palette_name],
        ),
        palette_name,
    )
    return palette_func


def color_to_rgb(color):
    """将一个颜色转化为 (0-255) 的 RGB tuple：'#858ACB'  --> (133, 138, 203)"""
    return color if isinstance(color, tuple) else tuple(int(x * 255) for x in to_rgb(color))


def gen_fa_mask(icon_name: str, size: int, icon_dir: str,invert_mask: bool, ttf_path: str = None, css_path: str = None):
    """
    这里的ttf_path也可以穿为icon_path
    将 Font Awesome icon 导出为 mask_array
    """
    icon_name_raw = icon_name.split(" ")[1]
    icon = IconFont(css_file=css_path, ttf_file=ttf_path)
    # 如果提供了长度和宽度，则将图标设为两者中较小的一个
    size = min(size) if isinstance(size, tuple) else size
    icon.export_icon(
        icon=icon_name_raw[len(icon.common_prefix):],
        size=size,
        filename="icon.png",
        export_dir=icon_dir,
    )
    icon = Image.open(os.path.join(icon_dir, "icon.png"))
    size = (size, size)
    w, h = icon.size
    icon_mask = Image.new("RGBA", icon.size, (255, 255, 255, 255))
    icon_mask.paste(icon, icon)
    mask = Image.new("RGBA", size, (255, 255, 255, 255))
    mask_w, mask_h = mask.size
    offset = ((mask_w - w) // 2, (mask_h - h) // 2)
    mask.paste(icon_mask, offset)
    mask_array = np.array(mask, dtype="uint8")
    mask_array = np.invert(mask_array) if invert_mask else mask_array

    return mask_array


def gen_pic_mask(pic_path: str, size: int, invert_mask: bool=False):
    """从图片生成一个mask_array"""
    pic = Image.open(pic_path)
    size = (size, size)
    w, h = pic.size
    icon_mask = Image.new("RGBA", pic.size, (255, 255, 255, 255))
    icon_mask.paste(pic, pic)
    mask = Image.new("RGBA", size, (255, 255, 255, 255))
    mask_w, mask_h = mask.size
    offset = ((mask_w - w) // 2, (mask_h - h) // 2)
    mask.paste(icon_mask, offset)
    mask_array = np.array(mask, dtype="uint8")
    mask_array = np.invert(mask_array) if invert_mask else mask_array
    return mask_array


def gen_gradient_mask( mask_array, size: int, palette: str, gradient_dir: str = "horizontal"):
    """从指定的 palette 生成一个 渐变色的 mask"""
    mask_array = np.float32(mask_array)
    palette_func = gen_palette(palette)
    gradient = palette_func.mpl_colormap(np.linspace(0.0, 1.0, size))
    # matplotlib color 映射的范围为 (0, 1). 转换为RGB
    gradient *= 255.0
    # 添加新轴并在其上重复渐变.
    gradient = np.tile(gradient, (size, 1, 1))
    # 如果是垂直的，则转置渐变.
    if gradient_dir == "vertical":
        gradient = np.transpose(gradient, (1, 0, 2))
    # 将图标上的任何非白色像素转换为渐变色.
    white = (255.0, 255.0, 255.0, 255.0)
    mask_array[mask_array != white] = gradient[mask_array != white]
    # 基于彩色图像的颜色生成器
    image_colors = ImageColorGenerator(mask_array)
    return image_colors, np.uint8(mask_array)


def gen_pal_colors(mask_array, size, colors, gradient, palette):
    if gradient and colors is None:
        # 如果需要设置渐变色，要求colors需为空且存在gradient字符串
        return gen_gradient_mask(mask_array, size, palette, gradient)
    else:
        if colors:
            # 判断colors是否为列表
            colors = [colors] if isinstance(colors, str) else colors
            colors = [color_to_rgb(color) for color in colors]
        else:
            palette_func = gen_palette(palette)
            colors = palette_func.colors
        def pal_colors(word, font_size, position, orientation, random_state, **kwargs):
            rand_color = np.random.randint(0, len(colors))
            return tuple(colors[rand_color])
        return pal_colors, mask_array


def gen_stylecloud(
        size: int = 512,
        icon_name: str = "fas fa-flag",
        palette: str = "cartocolors.qualitative.Bold_5",
        colors: Union[str, List[str]] = None,
        picpath: str = None,
        background_color: str = None,
        mode: str = "RGB",
        max_font_size: int = 200,
        min_font_size: int = 4,
        max_words: int = 200,
        icon_dir: str = ".temp",
        gradient: str = None,
        font_path: str = None,
        random_state: int = None,
        collocations: bool = True,
        invert_mask: bool = False,
        pro_icon_path: str = None,
        pro_css_path: str = None,
):
    """返回一个Word cloud对
    :param size: Size 长度和宽度（以像素为单位）
    :param icon_name: 图标名称. (e.g. 'fas fa-grin')
    :param palette: Color palette (via palettable)
    :param colors: 文本的自定义颜色(name or hex). 将覆盖palette
    :param picpath:图片路径，注意图片路径会和icon_name冲突
    :param background_color: 单词云图像的背景色(name or hex).
    :param mode:当模式为“RGBA”且background_color为None时，将生成透明背景。
    :param max_font_size: stylecloud中的最大字体大小.
    :param max_words: 要包含在stylecloud中的最大字数.
    :param icon_dir: 用于存储图标掩码图像的临时目录.
    :param gradient: 渐变方向
    :param font_path: 字体.ttf文件的路径
    :param random_state: 控制单词和颜色的随机状态
    :param collocations: 是否包括两个单词的搭配
    :param invert_mask: 是否反转图标掩码.
    :param pro_icon_path: Font Awesome的.ttf 路径.
    :param pro_css_path: Font Awesome 的.css 路径.
    """
    # 控制词图形状
    if picpath is None:
        mask_array = gen_fa_mask(icon_name, size, icon_dir, invert_mask, pro_icon_path, pro_css_path)
        rmtree(icon_dir)
    else:
        mask_array = gen_pic_mask(picpath, size, invert_mask)
    # 控制文字颜色
    color_func, mask_array = gen_pal_colors(mask_array, size, colors, gradient, palette)
    wc = WordCloud(
        background_color=background_color,
        mode=mode,
        font_path=font_path,
        max_words=max_words,
        mask=mask_array,
        max_font_size=max_font_size,
        min_font_size=min_font_size,
        random_state=random_state,
        collocations=collocations,
    )
    return color_func, wc


if __name__ == '__main__':
    maskarray = gen_fa_mask(
        icon_name='iconfont icon-nannv',
        size=512,
        icon_dir=".temp",
        invert_mask=False,
        ttf_path='./datas/fonts/iconfont.ttf',
        css_path='./datas/fonts/iconfont.css'
    )
    print(maskarray)
    # 尝试在这里创建人物的mask 图标









