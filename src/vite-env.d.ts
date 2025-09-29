/// <reference types="vite/client" />
/// <reference types="vite-plugin-pwa/client" />

interface ImportMetaEnv {
  readonly VITE_APP_TITLE: string;
  readonly PEERJS_HOST?: string;
  readonly PEERJS_PORT?: string;
  readonly PEERJS_PATH?: string;
  // more env variables...
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
