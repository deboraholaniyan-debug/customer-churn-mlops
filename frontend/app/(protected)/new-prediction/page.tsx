
"use client";

import { useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Loader2,
  RotateCcw,
  ShieldCheck,
  TrendingUp,
  UserRound,
} from "lucide-react";

type FormValues = {
  gender: string;
  SeniorCitizen: string;
  Partner: string;
  Dependents: string;
  tenure: string;
  PhoneService: string;
  MultipleLines: string;
  InternetService: string;
  OnlineSecurity: string;
  OnlineBackup: string;
  DeviceProtection: string;
  TechSupport: string;
  StreamingTV: string;
  StreamingMovies: string;
  Contract: string;
  PaperlessBilling: string;
  PaymentMethod: string;
  MonthlyCharges: string;
  TotalCharges: string;
};

type PredictionResult = {
  predicted_churn: "Yes" | "No";
  churn_probability: number;
  classification_threshold: number;
  model_name: string;
  model_version: string;
};

const initialValues: FormValues = {
  gender: "Female",
  SeniorCitizen: "0",
  Partner: "Yes",
  Dependents: "No",
  tenure: "12",
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "No",
  StreamingMovies: "No",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: "70",
  TotalCharges: "840",
};

const selectFields: {
  name: keyof FormValues;
  label: string;
  options: string[];
}[] = [
  {
    name: "gender",
    label: "Gender",
    options: ["Female", "Male"],
  },
  {
    name: "SeniorCitizen",
    label: "Senior citizen",
    options: ["0", "1"],
  },
  {
    name: "Partner",
    label: "Has a partner?",
    options: ["Yes", "No"],
  },
  {
    name: "Dependents",
    label: "Has dependents?",
    options: ["Yes", "No"],
  },
  {
    name: "PhoneService",
    label: "Phone service",
    options: ["Yes", "No"],
  },
  {
    name: "MultipleLines",
    label: "Multiple lines",
    options: ["Yes", "No", "No phone service"],
  },
  {
    name: "InternetService",
    label: "Internet service",
    options: ["DSL", "Fiber optic", "No"],
  },
  {
    name: "OnlineSecurity",
    label: "Online security",
    options: ["Yes", "No", "No internet service"],
  },
  {
    name: "OnlineBackup",
    label: "Online backup",
    options: ["Yes", "No", "No internet service"],
  },
  {
    name: "DeviceProtection",
    label: "Device protection",
    options: ["Yes", "No", "No internet service"],
  },
  {
    name: "TechSupport",
    label: "Technical support",
    options: ["Yes", "No", "No internet service"],
  },
  {
    name: "StreamingTV",
    label: "Streaming TV",
    options: ["Yes", "No", "No internet service"],
  },
  {
    name: "StreamingMovies",
    label: "Streaming movies",
    options: ["Yes", "No", "No internet service"],
  },
  {
    name: "Contract",
    label: "Contract type",
    options: ["Month-to-month", "One year", "Two year"],
  },
  {
    name: "PaperlessBilling",
    label: "Paperless billing",
    options: ["Yes", "No"],
  },
  {
    name: "PaymentMethod",
    label: "Payment method",
    options: [
      "Electronic check",
      "Mailed check",
      "Bank transfer (automatic)",
      "Credit card (automatic)",
    ],
  },
];

const numberFields: {
  name: keyof FormValues;
  label: string;
  placeholder: string;
  min?: number;
  max?: number;
  step?: string;
  helper?: string;
}[] = [
  {
    name: "tenure",
    label: "Customer tenure (months)",
    placeholder: "e.g. 12",
    min: 0,
    max: 72,
    helper: "Number of months the customer has stayed with the company.",
  },
  {
    name: "MonthlyCharges",
    label: "Monthly charges",
    placeholder: "e.g. 70.50",
    min: 0,
    step: "0.01",
    helper: "The customer's current monthly bill.",
  },
  {
    name: "TotalCharges",
    label: "Total charges",
    placeholder: "e.g. 840.00",
    min: 0,
    step: "0.01",
    helper: "The customer's accumulated charges.",
  },
];

export default function NewPredictionPage() {
  const [values, setValues] = useState<FormValues>(initialValues);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<PredictionResult | null>(null);

  function handleChange(name: keyof FormValues, value: string) {
    setValues((previous) => ({
      ...previous,
      [name]: value,
    }));
  }

  function resetForm() {
    setValues(initialValues);
    setResult(null);
    setError("");
  }

  async function handlePrediction(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    const payload = {
      ...values,
      SeniorCitizen: Number(values.SeniorCitizen),
      tenure: Number(values.tenure),
      MonthlyCharges: Number(values.MonthlyCharges),
      TotalCharges: Number(values.TotalCharges),
    };

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error || data.detail || "The prediction could not be completed."
        );
      }

      setResult(data as PredictionResult);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An unexpected error occurred. Please try again."
      );
    } finally {
      setLoading(false);
    }
  }

  const isHighRisk = result?.predicted_churn === "Yes";

  return (
    <main className="min-h-screen bg-slate-50 px-4 py-8 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-5xl">
        <div className="mb-8">
          <div className="flex items-center gap-2 text-sm font-medium text-blue-700">
            <TrendingUp size={17} />
            ChurnOps / Customer Prediction
          </div>

          <h1 className="mt-3 text-3xl font-bold tracking-tight text-slate-900">
            Predict Customer Churn
          </h1>

          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
            Enter the customer&apos;s account and service information to
            estimate their likelihood of leaving your organisation.
          </p>
        </div>

        <form onSubmit={handlePrediction} className="space-y-6">
          {/* Customer information */}
          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-6 flex items-center gap-3">
              <div className="rounded-xl bg-blue-50 p-2.5 text-blue-700">
                <UserRound size={21} />
              </div>

              <div>
                <h2 className="text-lg font-semibold text-slate-900">
                  Customer information
                </h2>
                <p className="text-sm text-slate-500">
                  Basic customer profile and relationship details.
                </p>
              </div>
            </div>

            <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
              {selectFields.slice(0, 4).map((field) => (
                <div key={field.name}>
                  <label
                    htmlFor={field.name}
                    className="mb-2 block text-sm font-medium text-slate-700"
                  >
                    {field.label}
                  </label>

                  <select
                    id={field.name}
                    value={values[field.name]}
                    onChange={(event) =>
                      handleChange(field.name, event.target.value)
                    }
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-800 outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                  >
                    {field.options.map((option) => (
                      <option key={option} value={option}>
                        {field.name === "SeniorCitizen"
                          ? option === "1"
                            ? "Yes"
                            : "No"
                          : option}
                      </option>
                    ))}
                  </select>
                </div>
              ))}

              {numberFields.slice(0, 1).map((field) => (
                <div key={field.name}>
                  <label
                    htmlFor={field.name}
                    className="mb-2 block text-sm font-medium text-slate-700"
                  >
                    {field.label}
                  </label>

                  <input
                    id={field.name}
                    type="number"
                    required
                    min={field.min}
                    max={field.max}
                    value={values[field.name]}
                    onChange={(event) =>
                      handleChange(field.name, event.target.value)
                    }
                    placeholder={field.placeholder}
                    className="w-full rounded-lg border border-slate-300 px-3 py-2.5 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                  />

                  <p className="mt-1 text-xs text-slate-500">{field.helper}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Telecommunications services */}
          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-slate-900">
              Telecommunications services
            </h2>
            <p className="mb-6 mt-1 text-sm text-slate-500">
              Specify the services currently used by the customer.
            </p>

            <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
              {selectFields.slice(4, 7).map((field) => (
                <div key={field.name}>
                  <label
                    htmlFor={field.name}
                    className="mb-2 block text-sm font-medium text-slate-700"
                  >
                    {field.label}
                  </label>

                  <select
                    id={field.name}
                    value={values[field.name]}
                    onChange={(event) =>
                      handleChange(field.name, event.target.value)
                    }
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                  >
                    {field.options.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </div>
              ))}
            </div>
          </section>

          {/* Additional services */}
          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-slate-900">
              Additional services and billing
            </h2>
            <p className="mb-6 mt-1 text-sm text-slate-500">
              Enter the customer&apos;s additional services, contract and
              billing details.
            </p>

            <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
              {selectFields.slice(7).map((field) => (
                <div key={field.name}>
                  <label
                    htmlFor={field.name}
                    className="mb-2 block text-sm font-medium text-slate-700"
                  >
                    {field.label}
                  </label>

                  <select
                    id={field.name}
                    value={values[field.name]}
                    onChange={(event) =>
                      handleChange(field.name, event.target.value)
                    }
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                  >
                    {field.options.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </div>
              ))}

              {numberFields.slice(1).map((field) => (
                <div key={field.name}>
                  <label
                    htmlFor={field.name}
                    className="mb-2 block text-sm font-medium text-slate-700"
                  >
                    {field.label}
                  </label>

                  <input
                    id={field.name}
                    type="number"
                    required
                    min={field.min}
                    step={field.step}
                    value={values[field.name]}
                    onChange={(event) =>
                      handleChange(field.name, event.target.value)
                    }
                    placeholder={field.placeholder}
                    className="w-full rounded-lg border border-slate-300 px-3 py-2.5 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                  />

                  <p className="mt-1 text-xs text-slate-500">{field.helper}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Error message */}
          {error && (
            <div
              role="alert"
              className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700"
            >
              <AlertCircle size={19} className="mt-0.5 shrink-0" />
              <p>{error}</p>
            </div>
          )}

          {/* Form actions */}
          <div className="flex flex-col gap-3 sm:flex-row sm:justify-end">
            <button
              type="button"
              onClick={resetForm}
              disabled={loading}
              className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <RotateCcw size={16} />
              Reset form
            </button>

            <button
              type="submit"
              disabled={loading}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-700 px-6 py-3 text-sm font-semibold text-white transition hover:bg-blue-800 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? (
                <>
                  <Loader2 size={17} className="animate-spin" />
                  Analysing customer...
                </>
              ) : (
                <>
                  <TrendingUp size={17} />
                  Predict churn
                </>
              )}
            </button>
          </div>
        </form>

        {/* Customer-friendly prediction result */}
        {result && (
          <section
            aria-live="polite"
            className="mt-8 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8"
          >
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500">
                  Prediction completed
                </p>
                <h2 className="mt-1 text-xl font-bold text-slate-900">
                  Customer churn assessment
                </h2>
              </div>

              <span
                className={`inline-flex w-fit items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold ${
                  isHighRisk
                    ? "bg-red-100 text-red-700"
                    : "bg-green-100 text-green-700"
                }`}
              >
                {isHighRisk ? (
                  <AlertCircle size={17} />
                ) : (
                  <CheckCircle2 size={17} />
                )}
                {isHighRisk ? "High churn risk" : "Low churn risk"}
              </span>
            </div>

            <div className="mt-8 grid gap-8 md:grid-cols-2 md:items-center">
              <div>
                <p className="text-sm font-medium text-slate-500">
                  Estimated churn probability
                </p>

                <p className="mt-2 text-5xl font-bold tracking-tight text-slate-900">
                  {(result.churn_probability * 100).toFixed(2)}%
                </p>

                <p className="mt-2 text-sm text-slate-500">
                  Estimated likelihood that this customer will leave.
                </p>
              </div>

              <div>
                <div className="mb-2 flex justify-between text-sm">
                  <span className="font-medium text-slate-600">
                    Churn probability
                  </span>
                  <span className="font-semibold text-slate-900">
                    {(result.churn_probability * 100).toFixed(2)}%
                  </span>
                </div>

                <div
                  className="h-4 overflow-hidden rounded-full bg-slate-100"
                  role="progressbar"
                  aria-label="Estimated churn probability"
                  aria-valuenow={result.churn_probability * 100}
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${
                      isHighRisk ? "bg-red-500" : "bg-green-500"
                    }`}
                    style={{
                      width: `${Math.min(
                        100,
                        Math.max(0, result.churn_probability * 100)
                      )}%`,
                    }}
                  />
                </div>

                <p className="mt-3 text-xs leading-5 text-slate-500">
                  The model uses a classification threshold of{" "}
                  {(result.classification_threshold * 100).toFixed(0)}% to
                  determine the predicted outcome.
                </p>
              </div>
            </div>

            <div
              className={`mt-8 rounded-xl p-5 ${
                isHighRisk
                  ? "border border-red-100 bg-red-50"
                  : "border border-green-100 bg-green-50"
              }`}
            >
              <div className="flex items-start gap-3">
                {isHighRisk ? (
                  <AlertCircle
                    size={21}
                    className="mt-0.5 shrink-0 text-red-600"
                  />
                ) : (
                  <CheckCircle2
                    size={21}
                    className="mt-0.5 shrink-0 text-green-600"
                  />
                )}

                <div>
                  <h3
                    className={`font-semibold ${
                      isHighRisk ? "text-red-800" : "text-green-800"
                    }`}
                  >
                    {isHighRisk
                      ? "Customer flagged as likely to churn"
                      : "Customer classified as likely to stay"}
                  </h3>

                  <p
                    className={`mt-1 text-sm leading-6 ${
                      isHighRisk ? "text-red-700" : "text-green-700"
                    }`}
                  >
                    {isHighRisk
                      ? "The model estimates that this customer has a churn probability at or above the configured classification threshold. Consider reviewing the customer's service experience and engagement."
                      : "The model estimates that this customer's churn probability is below the configured classification threshold. Continue monitoring the customer and maintaining service quality."}
                  </p>
                </div>
              </div>
            </div>

            <div className="mt-6 flex flex-col gap-3 border-t border-slate-100 pt-5 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <ShieldCheck size={16} />
                <span>
                  Model: {result.model_name} · Version {result.model_version}
                </span>
              </div>

              <button
                type="button"
                onClick={() => {
                  setResult(null);
                  setError("");
                  window.scrollTo({ top: 0, behavior: "smooth" });
                }}
                className="text-sm font-semibold text-blue-700 hover:text-blue-900"
              >
                Run another prediction
              </button>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}
