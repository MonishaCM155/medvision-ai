/**
 * config.ts — centralized environment configuration for the MedVision AI server.
 *
 * Every configurable value lives here and is read from environment variables
 * (see .env.example). Values are validated on startup via `validateConfig()`,
 * which returns human-readable problems instead of failing silently.
 */

import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';

dotenv.config();

const DEFAULT_PORT = 3000;

function validPort(n: number): number | null {
  return Number.isFinite(n) && n >= 1 && n <= 65535 ? Math.round(n) : null;
}

function portFromEnvFile(): number | null {
  try {
    const envPath = path.join(process.cwd(), '.env');
    if (fs.existsSync(envPath)) {
      const parsed = dotenv.parse(fs.readFileSync(envPath, 'utf8'));
      if (parsed.PORT && parsed.PORT.trim() !== '') return validPort(Number(parsed.PORT));
    }
  } catch {
    /* unreadable .env — ignore */
  }
  return null;
}

/**
 * Resolve the listen port, in priority order:
 *   1. PORT explicitly written into the project's .env file.
 *   2. process.env.PORT (Render/Heroku/Cloud Run assign one) — unless
 *      MEDVISION_IGNORE_ENV_PORT=1 (for runtimes that inject ephemeral ports).
 *   3. Project default 3000.
 */
function resolvePort(): number {
  const fromFile = portFromEnvFile();
  if (fromFile !== null) return fromFile;
  if (process.env.MEDVISION_IGNORE_ENV_PORT !== '1' && process.env.PORT) {
    const fromEnv = validPort(Number(process.env.PORT));
    if (fromEnv !== null) return fromEnv;
  }
  return DEFAULT_PORT;
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const LOG_LEVELS: LogLevel[] = ['debug', 'info', 'warn', 'error'];

function str(name: string, fallback: string): string {
  const v = process.env[name];
  return v === undefined || v.trim() === '' ? fallback : v.trim();
}

function int(name: string, fallback: number, min?: number, max?: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === '') return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n)) {
    console.warn(`[config] ${name}="${raw}" is not a number — using default ${fallback}`);
    return fallback;
  }
  if ((min !== undefined && n < min) || (max !== undefined && n > max)) {
    console.warn(`[config] ${name}="${raw}" is outside [${min ?? '-∞'}, ${max ?? '∞'}] — using default ${fallback}`);
    return fallback;
  }
  return Math.round(n);
}

function num(name: string, fallback: number, min?: number, max?: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === '') return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n)) return fallback;
  if ((min !== undefined && n < min) || (max !== undefined && n > max)) return fallback;
  return n;
}

export const APP_VERSION = '2.7.0';
export const APP_NAME = 'medvision-ai';

export const config = {
  nodeEnv: str('NODE_ENV', 'development'),
  port: resolvePort(),

  // AI / integration — PYTORCH_ENGINE_URL is the canonical name; FASTAPI_URL
  // is accepted as an alias.
  engineBase: str('PYTORCH_ENGINE_URL', str('FASTAPI_URL', 'http://127.0.0.1:8000')),
  geminiApiKey: str('GEMINI_API_KEY', ''),
  // Gemini model used for report + copilot synthesis. Overridable via env;
  // if the model id is invalid the request fails gracefully to the rule engine.
  geminiModel: str('GEMINI_MODEL', 'gemini-3.6-flash'),

  // Security (public research mode — no authentication, but abuse protection)
  apiRateLimit: int(
    'MEDVISION_API_RATE_LIMIT',
    int('MEDVISION_RATE_LIMIT', 60, 1, 100000),
    1,
    100000
  ),
  uploadMaxBytes: int('MAX_UPLOAD_SIZE', 27 * 1024 * 1024, 1024, 512 * 1024 * 1024),
  allowedOrigins: str('ALLOWED_ORIGINS', '')
    .split(',')
    .map((s) => s.trim().replace(/\/+$/, '')) // tolerate trailing slashes
    .filter(Boolean),

  // Reverse-proxy hop count for rate limiting / audit IPs. OFF by default —
  // only enable (TRUST_PROXY=1 or a hop count) when deployed behind Render,
  // nginx, or a cloud LB, so req.ip reflects the real client, not the proxy.
  trustProxy: (() => {
    const raw = str('TRUST_PROXY', '');
    if (raw === 'true') return true;
    const n = Number(raw);
    return Number.isInteger(n) && n > 0 ? n : false;
  })(),

  // AI safety pipeline
  temperature: num('MEDVISION_TEMPERATURE', 1.0, 0.2, 5.0),
  qualityThreshold: int('QUALITY_THRESHOLD', 55, 0, 100),
  confidenceThreshold: num('MEDVISION_CONFIDENCE_THRESHOLD', 0.75, 0.5, 0.95),

  // Persistence
  databaseUrl: str('DATABASE_URL', ''),

  // Observability
  logLevel: (() => {
    const lvl = str('LOG_LEVEL', 'info') as LogLevel;
    return LOG_LEVELS.includes(lvl) ? lvl : 'info';
  })(),
};

/**
 * Validate the loaded configuration and return a list of human-readable
 * problems. The caller decides whether to fail hard or warn.
 */
export function validateConfig(): string[] {
  const problems: string[] = [];

  if (!/^https?:\/\/.+/.test(config.engineBase)) {
    problems.push(`PYTORCH_ENGINE_URL "${config.engineBase}" is not a valid http(s) URL.`);
  }
  if (config.port < 1 || config.port > 65535) {
    problems.push(`PORT ${config.port} is outside the valid range 1-65535.`);
  }
  if (config.uploadMaxBytes < 1024) {
    problems.push('MAX_UPLOAD_SIZE is implausibly small (< 1 KB).');
  }
  return problems;
}
