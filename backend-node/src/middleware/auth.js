import jwt from 'jsonwebtoken';
import Guardian from '../models/Guardian.js';
import Patient from '../models/Patient.js';
import { AppError } from './errorHandler.js';

const DEV_FALLBACK_JWT_SECRET = 'auraos-dev-secret-change-before-production';
const getJwtSecret = () => {
  if (process.env.JWT_SECRET) return process.env.JWT_SECRET;

  if (process.env.NODE_ENV === 'production') {
    throw new AppError('JWT_SECRET must be configured in production.', 500);
  }

  return DEV_FALLBACK_JWT_SECRET;
};

export const signAuthToken = ({ id, role }) =>
  jwt.sign({ sub: String(id), role }, getJwtSecret(), { expiresIn: process.env.JWT_EXPIRES_IN || '7d' });

export const requireAuth = async (req, _res, next) => {
  const header = req.get('authorization') || '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : null;
  if (!token) return next(new AppError('Authentication required.', 401));

  try {
    const payload = jwt.verify(token, getJwtSecret());
    const Model = payload.role === 'guardian' ? Guardian : payload.role === 'patient' ? Patient : null;
    if (!Model) throw new AppError('Invalid auth role.', 401);

    const account = await Model.findById(payload.sub);
    if (!account) throw new AppError('Account not found.', 401);

    req.auth = {
      id: account._id.toString(),
      role: payload.role,
      account,
    };
    return next();
  } catch (err) {
    if (err instanceof AppError) return next(err);
    return next(new AppError('Invalid or expired token.', 401));
  }
};

export const requireRole = (...roles) => (req, _res, next) => {
  if (!req.auth) return next(new AppError('Authentication required.', 401));
  if (!roles.includes(req.auth.role)) return next(new AppError('Insufficient permissions.', 403));
  return next();
};
