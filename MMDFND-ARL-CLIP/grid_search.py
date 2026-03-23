import itertools
import subprocess
import os

def main():
    # 1. 定義網格搜索參數
    gammas = [0.2, 0.4, 0.6, 0.8]
    Ts = [2, 4, 6, 8]
    early_stops = [4, 6, 8]
    
    csv_filename = "grid_search_results.csv"
    
    # 產生所有參數組合 (4 * 4 * 3 = 48 組)
    param_combinations = list(itertools.product(gammas, Ts, early_stops))
    print(f"開始網格搜索，共有 {len(param_combinations)} 組參數組合需要測試。")
    print(f"測試結果將自動寫入 {csv_filename} 中。\n")

    # 2. 迴圈執行
    for idx, (g, t, es) in enumerate(param_combinations, 1):
        print(f"[{idx}/{len(param_combinations)}] ========== 正在訓練: gamma={g}, T={t}, early_stop={es} ==========")
        
        # 組裝命令行指令
        cmd = [
            "python", "main.py",
            "--gamma", str(g),
            "--T", str(t),
            "--early_stop", str(es),
            "--save_csv", csv_filename
        ]
        
        # 執行主程式 (如果發生 OOM 或崩潰，這裡的執行會中斷當前這組，但不影響 python 外部迴圈，非常安全)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[錯誤] 參數組合 gamma={g}, T={t}, early_stop={es} 執行失敗。繼續下一組...")
            continue
            
    print(f"\n網格搜索全部完成！請查看 {csv_filename} 取得所有測試數據。")

if __name__ == '__main__':
    main()