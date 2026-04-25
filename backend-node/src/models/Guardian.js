import mongoose from 'mongoose';

const GuardianAnswerSchema = new mongoose.Schema(
  {
    id: { type: String, required: true, maxlength: 80 },
    label: { type: String, default: '', maxlength: 180 },
    value: { type: Number, min: 0, max: 4, required: true },
  },
  { _id: false }
);

const DerivedGuardianScoresSchema = new mongoose.Schema(
  {
    observedIsolation: { type: Number, min: 0, max: 10, default: 0 },
    panicTriggerLoad: { type: Number, min: 0, max: 10, default: 0 },
    sleepConcern: { type: Number, min: 0, max: 10, default: 0 },
    routineConcern: { type: Number, min: 0, max: 10, default: 0 },
    supportFit: { type: Number, min: 0, max: 10, default: 0 },
    overallConcern: { type: Number, min: 0, max: 10, default: 0 },
  },
  { _id: false }
);

const GuardianIntakeSchema = new mongoose.Schema(
  {
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient', required: true },
    answers: { type: [GuardianAnswerSchema], default: [] },
    derivedScores: { type: DerivedGuardianScoresSchema, default: () => ({}) },
    completedAt: { type: Date, default: Date.now },
  },
  { _id: false }
);

const AlertPreferencesSchema = new mongoose.Schema(
  {
    whatsapp: { type: Boolean, default: true },
    sms: { type: Boolean, default: false },
    email: { type: Boolean, default: true },
    reportDownloads: { type: Boolean, default: true },
  },
  { _id: false }
);

const AuthMetaSchema = new mongoose.Schema(
  {
    lastLoginAt: { type: Date, default: null },
    passwordChangedAt: { type: Date, default: null },
  },
  { _id: false }
);

const GuardianSchema = new mongoose.Schema(
  {
    email: { type: String, required: true, unique: true, lowercase: true, trim: true, index: true },
    passwordHash: { type: String, required: true, select: false },
    displayName: { type: String, required: true, trim: true, maxlength: 120 },
    role: { type: String, enum: ['guardian'], default: 'guardian', immutable: true },
    linkedPatientIds: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Patient', index: true }],
    guardianIntakes: { type: [GuardianIntakeSchema], default: [] },
    alertPreferences: { type: AlertPreferencesSchema, default: () => ({}) },
    authMeta: { type: AuthMetaSchema, default: () => ({}) },
  },
  { timestamps: true }
);

GuardianSchema.index({ 'guardianIntakes.patientId': 1 });

const Guardian = mongoose.model('Guardian', GuardianSchema);
export default Guardian;
