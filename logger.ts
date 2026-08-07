/**
 * logger.ts — structured, leveled logging for the MedVision AI server.
 *
 * Emits single-line JSON records to stdout (info/debug/warn) and stderr
 * (error), with a request middleware that tags every request with an ID and
 * logs method/path/status/duration on completion.
 */

import crypto from 'crypto';
import type { Request, Response, NextFunction } from 'express';
import type { LogLevel } from './config';

const LEVELS: Record<LogLevel, number> = { debug: 10, info: 20, warn: 30, error: 40 };

let threshold = 20; // default: info

export function setLogLevel(level: LogLevel): void {
  threshold = LEVELS[level] ?? 20;
}

type Meta = Record<string, unknown>;

function emit(level: LogLevel, msg: string, meta: Meta = {}): void {
  if (LEVELS[level] < threshold) return;
  const line = JSON.stringify({ ts: new Date().toISOString(), level, msg, ...meta });
  if (level === 'error') process.stderr.write(line + '\n');
  else process.stdout.write(line + '\n');
}

export const log = {
  debug: (msg: string, meta?: Meta) => emit('debug', msg, meta),
  info: (msg: string, meta?: Meta) => emit('info', msg, meta),
  warn: (msg: string, meta?: Meta) => emit('warn', msg, meta),
  error: (msg: string, meta?: Meta) => emit('error', msg, meta),
};

declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace Express {
    interface Request {
      reqId?: string;
    }
  }
}

/**
 * Request logging middleware. Attaches an x-request-id (honouring an inbound
 * one for distributed tracing) and logs a structured completion record.
 */
export function requestLogger(): (req: Request, res: Response, next: NextFunction) => void {
  return (req, res, next) => {
    const reqId = (req.headers['x-request-id'] as string) || crypto.randomUUID();
    req.reqId = reqId;
    res.setHeader('x-request-id', reqId);
    const start = process.hrtime.bigint();
    res.on('finish', () => {
      // Skip high-frequency self-polls so the log is not dominated by them
      if (req.path === '/api/monitoring' || req.path === '/api/metrics') return;
      const ms = Number(process.hrtime.bigint() - start) / 1e6;
      log.info('request', {
        reqId,
        method: req.method,
        path: req.originalUrl,
        status: res.statusCode,
        ms: Math.round(ms * 10) / 10,
      });
    });
    next();
  };
}

/**
 * Security response headers (dependency-free Helmet subset). Note: microphone
 * is deliberately NOT restricted — the app's voice dictation/reading features
 * rely on the Web Speech API.
 */
export function securityHeaders(): (req: Request, res: Response, next: NextFunction) => void {
  return (_req, res, next) => {
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('Referrer-Policy', 'no-referrer');
    res.setHeader('X-XSS-Protection', '0'); // modern browsers: rely on CSP instead
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
    res.setHeader('Permissions-Policy', 'camera=(), geolocation=(), display-capture=()');
    next();
  };
}
