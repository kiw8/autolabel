"""
데이터셋 분할 스크립트 (트랙 #5)
==================================
목적: NAS의 corner/center 영상 폴더를 영상 단위로 train/val/test 분할하고
     Ultralytics 표준 구조로 심볼릭 링크를 생성한다.
입력: /nas03/1_EV_LABELING/corner/, /nas03/1_EV_LABELING/center/
출력: /home1/Jwson08/autolabel/dataset/  +  data.yaml
"""

# ===== [import] =====
import os                   # ← os.symlink()으로 심볼릭 링크 생성에 사용
import random               # ← 영상 폴더 셔플 시 재현 가능한 시드 고정에 사용
import argparse             # ← --sample 플래그 파싱에 사용
import yaml                 # ← data.yaml 파일 생성에 사용 (PyYAML)
from pathlib import Path    # ← 경로 연산을 문자열 대신 객체로 처리 (가독성·안전성)


# ===== [설정] =====
NAS_ROOT  = Path("/nas03/1_EV_LABELING")
# ← NAS 원본 데이터 루트. corner/와 center/ 가 이 아래에 있음

DATASET   = Path("/home1/Jwson08/autolabel/dataset")
# ← Ultralytics가 읽을 최종 데이터셋 루트. 이 아래에 images/labels 구조 생성

SEED      = 42
# ← random.shuffle의 시드값. 같은 값이면 누가 실행해도 동일한 분할 결과를 보장 (재현성)

RATIOS    = (0.7, 0.2, 0.1)
# ← train:val:test 비율. 영상 수에 곱해서 각 split 크기 결정

SAMPLE_N  = 20
# ← --sample 옵션 시 corner/center 각각에서 사용할 영상 수. 전체 242개 대신 40개로 빠른 검증


# ===== [함수 1] 영상 폴더 목록 수집 =====
def get_video_folders(source_dir: Path, sample: int = None) -> list[Path]:
    folders = sorted([
        d for d in source_dir.iterdir()
        if d.is_dir() and not d.name.startswith("@")
        # ← @eaDir 같은 Synology NAS 메타데이터 폴더를 제외. '@'로 시작하는 폴더는 모두 스킵
    ])
    if sample is not None:
        folders = folders[:sample]
        # ← --sample 시 정렬된 리스트 앞 N개만 자름. sorted() 덕에 항상 같은 N개가 선택됨
    return folders


# ===== [함수 2] 영상 단위 train/val/test 분할 =====
def split_folders(folders: list[Path], ratios: tuple, seed: int) -> tuple:
    rng = random.Random(seed)
    # ← 전역 random 상태를 오염시키지 않기 위해 독립적인 Random 인스턴스 생성

    shuffled = folders[:]
    # ← 원본 리스트를 보호하기 위해 얕은 복사(슬라이싱). 원본 순서는 유지

    rng.shuffle(shuffled)
    # ← 시드가 고정된 RNG로 셔플 → 실행마다 동일한 순서 보장

    n        = len(shuffled)
    n_train  = int(n * ratios[0])
    n_val    = int(n * ratios[1])
    # ← int()로 내림 처리. 나머지는 자동으로 test에 몰림 (총합 = n 보장)

    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val :]
    # ← 슬라이싱으로 세 구간 분리. 경계값이 겹치지 않으므로 데이터 누락·중복 없음

    return train, val, test


# ===== [함수 3] 심볼릭 링크 생성 =====
def make_symlinks(
    video_folders: list[Path],
    split_name: str,
    exclude_log: list,          # ← 추가: 제외된 파일 경로를 기록할 리스트 (main에서 공유)
) -> tuple[int, int, int]:
    img_dir = DATASET / "images" / split_name
    lbl_dir = DATASET / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_images  = 0   # 학습에 포함된 이미지 수 (라벨 있는 것만)
    n_excluded = 0  # 라벨 없어서 제외된 이미지 수
    # ← 변경: n_labels·n_missing 대신 n_images(포함)·n_excluded(제외)로 변수명 정리

    for video_dir in video_folders:
        img_files = sorted(video_dir.glob("*.jpg"))

        for img_path in img_files:
            lbl_path = video_dir / "labels" / "train" / f"{img_path.stem}.txt"

            if not lbl_path.exists():
                exclude_log.append(str(img_path))
                # ← 변경: 경고 출력 대신 리스트에 누적. main에서 한꺼번에 파일로 저장
                n_excluded += 1
                continue
                # ← 핵심 변경: 라벨 없는 이미지는 이미지 링크조차 만들지 않고 건너뜀
                #   NAS 원본 파일은 전혀 건드리지 않으므로 언제든 다시 포함시킬 수 있음

            # 라벨 있는 이미지만 이 아래 코드에 도달
            img_link = img_dir / img_path.name
            if not img_link.exists() and not img_link.is_symlink():
                os.symlink(img_path.resolve(), img_link)

            lbl_link = lbl_dir / lbl_path.name
            if not lbl_link.exists() and not lbl_link.is_symlink():
                os.symlink(lbl_path.resolve(), lbl_link)
            # ← 이미지·라벨 링크를 한 블록에서 같이 생성 → 항상 쌍이 보장됨

            n_images += 1

    return n_images, n_excluded


# ===== [메인] =====
def main():
    # --- 인자 파싱 ---
    parser = argparse.ArgumentParser(description="NAS 영상 데이터셋 train/val/test 분할")
    parser.add_argument(
        "--sample", action="store_true",
        help=f"corner/center 각 {SAMPLE_N}개 영상만 사용 (파이프라인 검증용)"
    )
    # ← store_true: 플래그만 있으면 True, 없으면 False. 값을 받지 않는 on/off 스위치
    args = parser.parse_args()

    sample_n = SAMPLE_N if args.sample else None
    if args.sample:
        print(f"[--sample 모드] corner/center 각 {SAMPLE_N}개 영상만 사용\n")

    # --- Step 1. 영상 폴더 수집 ---
    print("=" * 55)
    print("[Step 1] 영상 폴더 수집")
    corner_folders = get_video_folders(NAS_ROOT / "corner", sample_n)
    center_folders = get_video_folders(NAS_ROOT / "center", sample_n)
    print(f"  corner: {len(corner_folders):3d}개")
    print(f"  center: {len(center_folders):3d}개")
    print(f"  합계  : {len(corner_folders) + len(center_folders):3d}개")

    # --- Step 2. 영상 단위 분할 ---
    print(f"\n[Step 2] 영상 단위 분할 (seed={SEED}, 비율={RATIOS})")
    print("  ※ corner와 center를 각각 분할 후 합산 → 두 소스의 분포 비율 유지\n")

    c_tr, c_va, c_te = split_folders(corner_folders, RATIOS, SEED)
    e_tr, e_va, e_te = split_folders(center_folders, RATIOS, SEED)
    # ← corner(c_)와 center(e_)를 독립적으로 분할해서 합산.
    #   만약 전체를 한 번에 섞으면 우연히 한쪽이 특정 split에 몰릴 수 있음

    splits = {
        "train": c_tr + e_tr,
        "val":   c_va + e_va,
        "test":  c_te + e_te,
    }
    # ← dict로 묶어두면 이후 루프에서 split 이름과 폴더 리스트를 함께 순회 가능

    for name, folders in splits.items():
        n_c = sum(1 for f in folders if "corner" in f.parts)
        n_e = sum(1 for f in folders if "center" in f.parts)
        # ← Path.parts: 경로를 튜플로 분해. "corner"나 "center"가 경로 요소에 있는지 확인
        print(f"  {name:5s}: {len(folders):3d}개 영상  (corner={n_c}, center={n_e})")

    # --- Step 3. 심볼릭 링크 생성 ---
    print(f"\n[Step 3] 심볼릭 링크 생성 (라벨 없는 이미지는 제외)")
    print(f"  대상 디렉토리: {DATASET}\n")
    total_images = total_excluded = 0
    exclude_log = []
    # ← exclude_log: 제외된 이미지 절대 경로를 모으는 리스트.
    #   모든 split에서 공유해서 마지막에 파일 하나로 저장

    for name, folders in splits.items():
        n_img, n_excl = make_symlinks(folders, name, exclude_log)
        total_images   += n_img
        total_excluded += n_excl
        status = f"  (제외 {n_excl}개)" if n_excl else "  ✅"
        print(f"  {name:5s}: 이미지 {n_img:5d}개 포함{status}")

    print(f"\n  전체 포함: {total_images}개 / 제외: {total_excluded}개")

    # 제외 목록을 파일로 저장
    if exclude_log:
        log_path = DATASET / "excluded_no_label.txt"
        with open(log_path, "w") as f:
            f.write("\n".join(exclude_log) + "\n")
            # ← 제외된 이미지 경로를 한 줄씩 저장. NAS 원본 경로이므로 나중에 라벨 추가 후
            #   다시 포함시키려면 이 파일을 참고하면 됨
        print(f"  제외 목록 저장: {log_path}  ({total_excluded}개)")

    # --- Step 4. data.yaml 생성 ---
    print(f"\n[Step 4] data.yaml 생성")
    yaml_path = DATASET / "data.yaml"
    data_yaml = {
        "path":  str(DATASET),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "names": {0: "person", 1: "dog"},
    }
    # ← Ultralytics는 path 기준 상대 경로로 images/labels 를 찾음
    # ← names를 dict로 쓰면 yaml.dump가 {0: person, 1: dog} 형태로 출력 (YOLO 표준 호환)

    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        # ← default_flow_style=False: 블록 스타일로 출력 (사람이 읽기 편한 형태)
        # ← sort_keys=False: 작성한 키 순서 그대로 유지 (path → train → val → test → names)

    print(f"  저장: {yaml_path}")
    print(f"\n{'=' * 55}")
    print("완료. 다음 명령으로 학습을 시작할 수 있습니다:")
    print(f"  yolo segment train data={yaml_path} model=yolo11n-seg.pt")


if __name__ == "__main__":
    main()
