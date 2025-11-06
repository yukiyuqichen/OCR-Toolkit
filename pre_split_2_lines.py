import os
import numpy as np
import cv2
from sklearn.cluster import KMeans  # 保留以兼容你原始环境（本版未使用）
import load_files


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def pre_split(
    pre_select_dir_path,
    pre_save_dir_path,
    frame_pre,
    process,
    file_num,
    count_num,
    # ▼▼ 可调冗余参数 ▼▼
    overlap_px: int = 0,
    overlap_ratio: float = 0.00,
    # ▼▼ ROI与直线检测参数 ▼▼
    roi_abs_width: int | None = None,
    roi_ratio: float = 0.15,           # ROI 相对宽度（随后以 1/3、2/3 为中心对称扩展）
    min_vertical_len_ratio: float = 0.05,  # HoughP 最小线长（相对图高）
    vertical_dx_tolerance: int = 15,   # 近竖线判定的 |dx| 容差
    # ▼▼ 调试开关 ▼▼
    debug: bool = False
):
    """
    根据中间界栏分割页面，并为左右中三段各加入可调冗余重叠。
    - ROI 以 width/3 与 2*width/3 为中心，向两侧对称扩展 roi_w/2；
    - 不采用垂直投影兜底；
    - 检测阶段：自适应阈值 + 竖向闭运算 + 自动Canny + HoughP；
    - 选线阶段：去掉 gapless 严格检查；综合评分改为“更靠近 1/3 / 2/3 中心线更好”；
    - debug=True 时分别输出：
        *_lines.png  —— “检测到的直线”可视化（黄框：ROI；灰：全部Hough线；蓝：近竖线候选）
        *_cuts.png   —— “最终用于切割的直线”可视化（绿：被选中线段；红：切割与冗余边界）
        以及中间产物：_gray/_binary/_binary_closed/_edges/_roi_left/_roi_right
    """

    # -------------------------------
    # 内部工具函数
    # -------------------------------
    def ensure_bgr(img):
        # 将灰度/带alpha统一为BGR，便于画彩色调试图
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def auto_threshold(gray):
        """更稳健的二值化：自适应阈值 + 反色（黑线白底 -> 前景255）"""
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 51, 10)
        return binary

    def morph_close_vertical(binary):
        """竖向闭运算，连通断裂的竖线"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    def auto_canny(img_binary):
        """基于中位数的自动Canny阈值"""
        v = np.median(img_binary)
        low = int(max(0, 0.33 * v))
        high = int(min(255, 1.33 * v))
        return cv2.Canny(img_binary, low, high, apertureSize=3)

    def try_hough_scan(edges, h):
        """参数扫描器（仅调试打印Top组合）"""
        results = []
        for th in (40, 60, 80):
            for mlen in (int(0.08*h), int(0.12*h), int(0.18*h)):
                for mgap in (10, 20, 30):
                    lines = cv2.HoughLinesP(edges, 1.0, np.pi/360, th,
                                            minLineLength=mlen, maxLineGap=mgap)
                    n = 0 if lines is None else len(lines)
                    results.append((th, mlen, mgap, n))
        results.sort(key=lambda x: -x[3])
        return results[:5]

    def keep_vertical(lines, dx_tol):
        """仅保留近竖线"""
        kept = []
        if lines is None:
            return kept
        for L in lines:
            x1, y1, x2, y2 = L[0]
            if abs(x2 - x1) <= dx_tol:
                kept.append(L)
        return kept

    def line_center_x(L):
        x1, y1, x2, y2 = L[0]
        return 0.5 * (x1 + x2)

    def line_length(L):
        x1, y1, x2, y2 = L[0]
        return float(np.hypot(x2 - x1, y2 - y1))

    def line_verticality(L):
        """竖直度：tanθ = |dx| / (|dy|+1e-6)，越小越竖"""
        x1, y1, x2, y2 = L[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1) + 1e-6
        return dx / dy

    # === 改动点1：评分改为“靠近中心线(1/3或2/3)更好” ===
    def score_line_centered(L, roi_side, width):
        """
        综合评分（新版）：
        - 竖直度（越竖越好）
        - 线段长度（越长越好）
        - 靠近 ROI 中心线（左=width/3；右=2*width/3）
        """
        vx = line_verticality(L)      # 越小越好
        ln = line_length(L)           # 越大越好
        cx = line_center_x(L)

        center_x = (width / 3.0) if roi_side == 'left' else (width * 2.0 / 3.0)
        center_dist = abs(cx - center_x)  # 越小越好

        v_score  = 1.0 / (1.0 + vx)       # 0~1
        ln_score = np.log1p(ln)           # 对数压缩
        ct_score = 1.0 / (1.0 + center_dist)

        # 权重：保持与旧版一致的风格（你可按需要微调）
        return 0.5 * v_score + 0.3 * ln_score + 0.2 * ct_score

    # === 改动点2：pick函数内移除了 gapless 检查 ===
    def pick_boundary_from_lines(lines, roi_side, width, height, dx_tol):
        """
        从候选线中选择最佳一条：
        - 竖直过滤（|dx|<=dx_tol）
        -（已移除）gapless 严格检查
        - 综合打分（靠近 1/3 或 2/3 中心线更好）
        返回：boundary_x(基于上下延长求交点后取均值), chosen_line(L or None)
        """
        verticals = keep_vertical(lines, dx_tol)
        if not verticals:
            return None, None

        # 综合评分选择最佳
        scores = [(score_line_centered(L, roi_side, width), L) for L in verticals]
        scores.sort(key=lambda t: -t[0])
        best_line = scores[0][1]

        # 外推求上下边界交点，再取均值x
        x1, y1, x2, y2 = best_line[0]
        eps = 1e-6
        point_top_x  = ((0 - y1) * (x2 - x1) / (y2 - y1 + eps)) + x1
        point_down_x = ((height - 1 - y1) * (x2 - x1) / (y2 - y1 + eps)) + x1
        boundary_x = (point_top_x + point_down_x) / 2.0

        return boundary_x, best_line

    # -------------------------------
    # 主流程
    # -------------------------------
    process.set("")
    frame_pre.update()

    filelist = load_files.load_img(pre_select_dir_path)
    file_len = len(filelist)
    file_num.set(file_len)
    count = 0
    count_num.set(0)

    file_path = pre_select_dir_path.get().rstrip("/\\")
    save_path = pre_save_dir_path.get().rstrip("/\\")
    os.makedirs(save_path, exist_ok=True)

    for file in filelist:
        base = str(file)
        in_path = os.path.join(file_path, base)

        # ---- 读取 & 预处理 ----
        img = cv2.imdecode(np.fromfile(in_path, dtype=np.uint8), -1)
        if img is None:
            count += 1
            count_num.set(count)
            continue

        img_bgr = ensure_bgr(img)

        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = img_bgr if img_bgr.ndim == 2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        binary = auto_threshold(gray)              # 前景=255
        binary_closed = morph_close_vertical(binary)
        edges = auto_canny(binary_closed)

        height, width = edges.shape[:2]
        h = height - 1
        w = width - 1

        # 两张独立调试底图
        debug_lines_img = img_bgr.copy()  # 仅“检测到的直线”
        debug_cuts_img  = img_bgr.copy()  # 仅“最终用于切割的直线”

        # 冗余宽度
        overlap = max(int(overlap_px), int(width * float(overlap_ratio)))

        # ===== ROI（以 1/3、2/3 为中心对称扩展）=====
        roi_w  = max(int(roi_ratio * width), int(roi_abs_width or 0))
        roi_w  = _clamp(roi_w, 20, max(60, width // 2))
        roi_hw = roi_w // 2

        cx_left  = int(round(width / 3.0))
        cx_right = int(round(width * 2.0 / 3.0))

        w1_1 = _clamp(cx_left  - roi_hw, 0, width)
        w1_2 = _clamp(cx_left  + roi_hw, 0, width)

        w2_1 = _clamp(cx_right - roi_hw, 0, width)
        w2_2 = _clamp(cx_right + roi_hw, 0, width)

        # 避免两ROI重叠（可选）
        if w1_2 >= w2_1:
            mid = (w1_2 + w2_1) // 2
            w1_2 = mid
            w2_1 = mid

        # 掩膜
        mask1 = np.zeros_like(edges)
        cv2.fillPoly(mask1, [np.array([(w1_1, height), (w1_2, height), (w1_2, 0), (w1_1, 0)], dtype=np.int32)], 255)
        masked_img1 = cv2.bitwise_and(edges, mask1)

        mask2 = np.zeros_like(edges)
        cv2.fillPoly(mask2, [np.array([(w2_1, height), (w2_2, height), (w2_2, 0), (w2_1, 0)], dtype=np.int32)], 255)
        masked_img2 = cv2.bitwise_and(edges, mask2)

        # ROI框只画在“检测到的直线”图上
        if debug:
            cv2.rectangle(debug_lines_img, (w1_1, 0), (w1_2, h), (0, 255, 255), 2)  # 左ROI（中心1/3）
            cv2.rectangle(debug_lines_img, (w2_1, 0), (w2_2, h), (0, 255, 255), 2)  # 右ROI（中心2/3）

        # 全图Hough扫描（仅打印）
        if debug:
            top = try_hough_scan(edges, height)
            print(f"[{base}] Top Hough combos (threshold, minLen, maxGap, count):", top)

        # 霍夫直线（更稳参数）
        min_line_len = int(max(20, height * float(min_vertical_len_ratio)))
        lines1 = cv2.HoughLinesP(masked_img1, 1.0, np.pi / 360, 60,
                                 minLineLength=min_line_len, maxLineGap=10)
        lines2 = cv2.HoughLinesP(masked_img2, 1.0, np.pi / 360, 60,
                                 minLineLength=min_line_len, maxLineGap=10)

        # 画出“所有检测到的直线”（灰色，画在 _lines.png）
        if debug:
            if lines1 is not None:
                for L in lines1:
                    x1, y1, x2, y2 = L[0]
                    cv2.line(debug_lines_img, (x1, y1), (x2, y2), (180, 180, 180), 1)
            if lines2 is not None:
                for L in lines2:
                    x1, y1, x2, y2 = L[0]
                    cv2.line(debug_lines_img, (x1, y1), (x2, y2), (180, 180, 180), 1)
            print(f"[{base}] L all lines:", 0 if lines1 is None else len(lines1),
                  "R all lines:", 0 if lines2 is None else len(lines2))

        # 仅保留“竖线”（蓝色，画在 _lines.png）
        vertical_lines1 = keep_vertical(lines1, vertical_dx_tolerance)
        vertical_lines2 = keep_vertical(lines2, vertical_dx_tolerance)
        if debug:
            for L in vertical_lines1:
                x1, y1, x2, y2 = L[0]
                cv2.line(debug_lines_img, (x1, y1), (x2, y2), (255, 0, 0), 1)  # 蓝色
            for L in vertical_lines2:
                x1, y1, x2, y2 = L[0]
                cv2.line(debug_lines_img, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # ---- 估计左右边界（新版：无gapless，中心化评分）----
        left_boundary_x, left_chosen_seg = pick_boundary_from_lines(
            lines1, 'left', width, height, vertical_dx_tolerance
        )
        right_boundary_x, right_chosen_seg = pick_boundary_from_lines(
            lines2, 'right', width, height, vertical_dx_tolerance
        )

        # 若未检测到，默认位置
        if left_boundary_x is None:
            left_boundary_x = width / 3.0
        if right_boundary_x is None:
            right_boundary_x = width * 2.0 / 3.0

        # 顺序 & 取整
        if left_boundary_x > right_boundary_x:
            left_boundary_x, right_boundary_x = right_boundary_x, left_boundary_x
            left_chosen_seg, right_chosen_seg = right_chosen_seg, left_chosen_seg

        Lx = int(round(left_boundary_x))
        Rx = int(round(right_boundary_x))
        Lx = _clamp(Lx, 0, w)
        Rx = _clamp(Rx, 0, w)

        # ====== 带冗余的切分范围 ======
        L_keep_end   = _clamp(Lx + overlap, 0, w)     # 左图保留到这
        M_keep_start = _clamp(Lx - overlap, 0, w)     # 中图起点
        M_keep_end   = _clamp(Rx + overlap, 0, w)     # 中图终点
        R_keep_start = _clamp(Rx - overlap, 0, w)     # 右图起点

        # ====== 生成三段图像 ======
        left_img = np.copy(img_bgr)
        poly_right_of_left = np.array([[(L_keep_end, 0), (w, 0), (w, h), (L_keep_end, h)]], dtype=np.int32)
        cv2.fillConvexPoly(left_img, poly_right_of_left, (255, 255, 255))

        middle_img = np.copy(img_bgr)
        poly_left_of_mid  = np.array([[(0, 0), (M_keep_start, 0), (M_keep_start, h), (0, h)]], dtype=np.int32)
        poly_right_of_mid = np.array([[(M_keep_end, 0), (w, 0), (w, h), (M_keep_end, h)]], dtype=np.int32)
        cv2.fillConvexPoly(middle_img, poly_left_of_mid,  (255, 255, 255))
        cv2.fillConvexPoly(middle_img, poly_right_of_mid, (255, 255, 255))

        right_img = np.copy(img_bgr)
        poly_left_of_right = np.array([[(0, 0), (R_keep_start, 0), (R_keep_start, h), (0, h)]], dtype=np.int32)
        cv2.fillConvexPoly(right_img, poly_left_of_right, (255, 255, 255))

        # ====== 保存图像 ======
        def _safe_path(suffix):
            return os.path.join(save_path, base + suffix)

        out_left   = _safe_path("_left.png")
        out_middle = _safe_path("_middle.png")
        out_right  = _safe_path("_right.png")
        cv2.imencode('.png', left_img)[1].tofile(out_left)
        cv2.imencode('.png', middle_img)[1].tofile(out_middle)
        cv2.imencode('.png', right_img)[1].tofile(out_right)

        # ====== Debug：分别保存“检测到的直线图”和“最终切割图”以及中间产物 ======
        if debug:
            # （A）最终切割图 —— 只画最终用于切割的元素
            if left_chosen_seg is not None:
                x1, y1, x2, y2 = left_chosen_seg[0]
                cv2.line(debug_cuts_img, (x1, y1), (x2, y2), (0, 220, 0), 3)
            if right_chosen_seg is not None:
                x1, y1, x2, y2 = right_chosen_seg[0]
                cv2.line(debug_cuts_img, (x1, y1), (x2, y2), (0, 220, 0), 3)

            def vline(img, x, color=(0, 0, 255), thick=2, label: str | None = None):
                xi = int(x)
                cv2.line(img, (xi, 0), (xi, h), color, thick)
                if label is not None:
                    cv2.putText(img, f"{label}:{xi}", (xi + 3, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            vline(debug_cuts_img, Lx, (0, 0, 255), 2, "L")
            vline(debug_cuts_img, Rx, (0, 0, 255), 2, "R")
            vline(debug_cuts_img, L_keep_end,   (0, 0, 255), 1, "L_end")
            vline(debug_cuts_img, M_keep_start, (0, 0, 255), 1, "M_start")
            vline(debug_cuts_img, M_keep_end,   (0, 0, 255), 1, "M_end")
            vline(debug_cuts_img, R_keep_start, (0, 0, 255), 1, "R_start")

            # （B）检测到的直线图 —— 前面已画ROI框、灰线、蓝线
            out_lines = _safe_path("_lines.png")
            out_cuts  = _safe_path("_cuts.png")
            cv2.imencode('.png', debug_lines_img)[1].tofile(out_lines)
            cv2.imencode('.png', debug_cuts_img)[1].tofile(out_cuts)

            # # 中间产物
            # out_gray   = _safe_path("_gray.png")
            # out_bin    = _safe_path("_binary.png")
            # out_bincl  = _safe_path("_binary_closed.png")
            # out_edges  = _safe_path("_edges.png")
            # out_roi_l  = _safe_path("_roi_left.png")
            # out_roi_r  = _safe_path("_roi_right.png")


        # 进度
        count += 1
        count_num.set(count)
        process.set(f"Processing: {count_num.get()} / {file_num.get()}")
        frame_pre.update()
        print(f"Finish: {count} / {file_len}")

    process.set("Finished!")
    frame_pre.update()
    print("Finished!")
