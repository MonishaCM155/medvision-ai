/**
 * storage.ts — durable persistence for MedVision AI.
 *
 * Backend hierarchy (first that works wins):
 *   1. postgres — when DATABASE_URL is set AND the optional `pg` package is
 *      installed (documented dependency, not required to boot).
 *   2. json     — zero-dependency JSON file under ./data/medvision-store.json.
 *   3. memory   — fallback (fresh every boot).
 *
 * The store persists the inference history and the security audit log.
 */

import fs from 'fs';
import path from 'path';
import { log } from './logger';

export interface AuditEntry {
  id: string;
  time: string;
  method: string;
  path: string;
  status: number;
  ip: string;
  /** Always "anonymous" — MedVision AI is a public platform with no accounts. */
  actor: string;
  detail?: string;
}

const STORE_DIR = path.join(process.cwd(), 'data');
const STORE_PATH = path.join(STORE_DIR, 'medvision-store.json');

let backend: 'postgres' | 'json' | 'memory' = 'memory';
let pgPool: any = null;
let jsonCache: { history: any[]; audit: AuditEntry[] } = { history: [], audit: [] };
let writeTimer: NodeJS.Timeout | null = null;

function readJsonFile(): { history: any[]; audit: AuditEntry[] } | null {
  try {
    if (!fs.existsSync(STORE_PATH)) return null;
    const parsed = JSON.parse(fs.readFileSync(STORE_PATH, 'utf8'));
    return {
      history: Array.isArray(parsed.history) ? parsed.history : [],
      audit: Array.isArray(parsed.audit) ? parsed.audit : [],
    };
  } catch (err) {
    log.warn('storage-read-failed', { error: (err as Error).message });
    return null;
  }
}

function scheduleWrite() {
  if (backend !== 'json') return;
  if (writeTimer) clearTimeout(writeTimer);
  writeTimer = setTimeout(() => {
    try {
      fs.mkdirSync(STORE_DIR, { recursive: true });
      fs.writeFileSync(STORE_PATH, JSON.stringify(jsonCache, null, 2), 'utf8');
    } catch (err) {
      log.warn('storage-write-failed', { error: (err as Error).message });
    }
  }, 500);
}

/**
 * Load persisted state into the given live arrays. Must be called once at boot.
 */
export async function initStorage(historyRef: any[], auditRef: AuditEntry[]): Promise<void> {
  const dbUrl = process.env.DATABASE_URL;
  if (dbUrl) {
    try {
      // Optional dependency — never a hard requirement (ambient-typed so the
      // project typechecks without installing the package)
      // @ts-ignore -- 'pg' is optional; only imported when DATABASE_URL is set
      const pg = await import('pg');
      const pool = new pg.Pool({ connectionString: dbUrl, connectionTimeoutMillis: 3000 });
      await pool.query('SELECT 1');
      await pool.query(
        `CREATE TABLE IF NOT EXISTS predictions (id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT now())`
      );
      await pool.query(
        `CREATE TABLE IF NOT EXISTS audit_logs (id TEXT PRIMARY KEY, time TEXT, method TEXT, path TEXT, status INT, ip TEXT, role TEXT, detail TEXT)`
      );
      pgPool = pool;
      backend = 'postgres';

      const preds = await pool.query('SELECT payload FROM predictions ORDER BY created_at DESC LIMIT 200');
      historyRef.length = 0;
      for (const row of preds.rows) {
        try {
          historyRef.push(JSON.parse(row.payload));
        } catch {
          /* skip corrupt row */
        }
      }
      const auditRes = await pool.query(
        'SELECT id, time, method, path, status, ip, role AS actor, detail FROM audit_logs ORDER BY time DESC LIMIT 500'
      );
      auditRef.length = 0;
      auditRef.push(...auditRes.rows);
      log.info('storage-postgres-connected', { history: historyRef.length, audit: auditRef.length });
      return;
    } catch (err) {
      log.warn('storage-postgres-unavailable', { error: (err as Error).message });
      pgPool = null;
    }
  }

  const saved = readJsonFile();
  if (saved) {
    jsonCache = saved;
    historyRef.length = 0;
    historyRef.push(...saved.history);
    auditRef.length = 0;
    auditRef.push(...saved.audit);
    backend = 'json';
    log.info('storage-json-loaded', { history: saved.history.length, audit: saved.audit.length, path: STORE_PATH });
  } else {
    jsonCache = { history: [...historyRef], audit: [...auditRef] };
    backend = 'json';
  }
}

export async function persistHistory(historyRef: any[]): Promise<void> {
  if (backend === 'postgres' && pgPool) {
    for (const item of historyRef.slice(0, 20)) {
      try {
        await pgPool.query(
          `INSERT INTO predictions (id, payload) VALUES ($1, $2) ON CONFLICT (id) DO NOTHING`,
          [item.id, JSON.stringify(item)]
        );
      } catch {
        /* best effort */
      }
    }
    return;
  }
  if (backend === 'json') {
    jsonCache.history = historyRef;
    scheduleWrite();
  }
}

export async function persistAudit(entry: AuditEntry): Promise<void> {
  if (backend === 'postgres' && pgPool) {
    try {
      // The Postgres column keeps its historical name `role` for schema
      // backward-compatibility; it always stores the anonymous actor.
      await pgPool.query(
        `INSERT INTO audit_logs (id, time, method, path, status, ip, role, detail) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`,
        [entry.id, entry.time, entry.method, entry.path, entry.status, entry.ip, entry.actor, entry.detail || null]
      );
    } catch {
      /* best effort */
    }
    return;
  }
  if (backend === 'json') {
    jsonCache.audit = [entry, ...jsonCache.audit].slice(0, 500);
    scheduleWrite();
  }
}

export function storageBackend(): string {
  return backend;
}
