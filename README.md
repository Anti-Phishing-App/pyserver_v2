# pyserver_v2
서버 구조 리팩터링을 위한 FastAPI pyserver v2 레포지토리입니다.

## 운영 원칙 (중요)
- 배포 시 Compose 파일은 루트의 `docker-compose.yml`만 사용합니다.
- `api_server`와 `ai_server`는 각각 `8000`, `8001` 포트를 사용합니다.
- 배포 후에는 헬스체크를 반드시 실행합니다.

## 배포 절차
```bash
cd /opt/apps/phishing-api/pyserver
git fetch v2
git pull --ff-only v2 main
docker compose down --remove-orphans
docker compose up -d --build --force-recreate
```

## 배포 후 점검
```bash
bash scripts/post_deploy_check.sh
```

스크립트는 다음을 자동 확인합니다.
- `api_server`, `ai_server` 컨테이너 실행 상태
- `http://127.0.0.1:8000/healthz` 응답
- `http://127.0.0.1:8001/healthz` 응답
