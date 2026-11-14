import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const apiUrl = process.env.CHURN_API_URL;
    const apiKey = process.env.CHURN_API_KEY;

    if (!apiUrl || !apiKey) {
      return NextResponse.json(
        { error: "Prediction service is not configured." },
        { status: 500 }
      );
    }

    const customerData = await request.json();

    const response = await fetch(`${apiUrl}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": apiKey,
      },
      body: JSON.stringify(customerData),
      cache: "no-store",
    });

    const result = await response.json();

    return NextResponse.json(result, {
      status: response.status,
    });
  } catch (error) {
    console.error("Prediction proxy error:", error);

    return NextResponse.json(
      { error: "Unable to communicate with the prediction service." },
      { status: 500 }
    );
  }
}