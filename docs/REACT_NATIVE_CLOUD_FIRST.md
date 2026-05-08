# React Native Cloud-First App (Cactus-aligned)

This repo now includes a React Native app scaffold at:

- `apps/mobile-rn`

## Stack

- React Native CLI (bare)
- TypeScript
- `cactus-react-native`
- `react-native-nitro-modules`
- `@react-native-documents/picker`

## Current MVP Flow

Implemented in `apps/mobile-rn/App.tsx`:

1. Enter cloud endpoint URL (your deployed `/v1/translate-sign` endpoint)
2. Select a video clip from device
3. Upload via multipart form (`video` field)
4. Display success fields (`gloss`, `translation`, `confidence`, `latency_ms`, `request_id`) or structured error

## Run locally

```bash
cd apps/mobile-rn
npm install --include=dev
cd ios && bundle install && bundle exec pod install && cd ..
npm run ios
```

For Android:

```bash
cd apps/mobile-rn
npm run android
```

## Endpoint note

Do **not** use `127.0.0.1` on a physical phone.
Use a reachable LAN or cloud URL such as:

- `http://192.168.x.x:8000/v1/translate-sign` (temporary local server)
- `https://<your-domain>/v1/translate-sign` (deployed backend)

## Next migration slices

- Replace Swift UI flow with RN navigation/screens (`Select -> Upload -> Result`)
- Add endpoint persistence (secure storage/settings screen)
- Add iOS/Android camera capture path and <=5s client-side duration validation
- Integrate real Cactus SDK runtime path toggle (cloud default, local optional)
- Decommission Swift app target after parity proof on device
