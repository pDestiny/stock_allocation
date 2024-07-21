# Modern Portfolio Theory(MPT) 기반 자본 분배 비율 계산 모델

## 사용법

### 파이썬 설치
[이 페이지 참조](https://wikidocs.net/8)

### 코드 다운로드
github에서 직접 zip 파일로 받는 방법도 있으며 git을 안다면 `git clone` 명령어로 다운받는 방법도 있다.
zip 파일로 받았다면 zip 파일을 풀고 코드가 있는 폴더로 들어간다.
그런다음 필요한 페키지를 pip 명령어를 이용해 requirements.txt 파일에 있는 패키지들을 설치한다.

```shell
pip install -r requirements.txt
```

패키지의 설치가 끝났다면 아래와 run.py 파일을 실행시켜 작동 시킬 수 있다.
```shell
python run.py --stock-ids 005930 000660 --start 2022-01-01 --end 2024-07-15 --days 252 --resolution 100
```

인자값들은 아래와 같다.
* --stock-ids : 한국 주식 cope를 의미한다. 예시에 적혀 있는 것은 삼성전자와 SK hynix 의 주식 아이디이다. 네이버 증권에 들어가면 원하는 주식의 code를 쉽게 얻을 수 있다.
* --start : 기대 수익률과 표준편차의 계산에 쓰일 기간중 시작 기간을 의미한다.
* --end : 기대 수익률과 표준편차의 계산에 쓰일 기간 중 끝 기간을 의미한다.
* --days : 기대 수익률과 표준편차는 일단위로 계산 된 다음에 --days 옵션에 지정한 기간으로 곱해진다.
* --resolution : efficient frontier를 계산 할 때, 목표 이익률을 얼마나 세밀하게 정할지를 결정한다. 예를 들면 목표 이익률 구간이 0 과 0.05 사이고 resolution이 5라면, 0, 0.01, 0.02, 0.03, 0.04 로 5개로 나뉠 것이고 10일 경우 0, 0.005, 0.01, 0.015 ...0.045 와 같이 10개로 나뉠 것이다.
* --outdir : 프로그램이 출력할 결과물을 저장할 폴더를 지정한다.

코드를 실행하면 이미지와 엑셀 파일이 출력된다. 이미지는 efficient frontier 라인을 그래프로 그린 것이고, excel 파일은 resolution에 따라 목표 수익률에 따른 표준 편차와 각 주식의 비율을 출력한다.



