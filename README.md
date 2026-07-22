# Assignment 1: Dataset Exploration & Problem Framing

## Dataset Overview
**Dataset:** [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

The dataset contains booking-level records for a City Hotel and a Resort Hotel in Portugal
(2015–2017), including reservation details (lead time, length of stay, arrival date), guest
details (adults/children/babies, country, repeat-guest status), and commercial details
(market segment, distribution channel, deposit type, average daily rate, number of special
requests). Each row is one booking, and one column (`is_canceled`) records whether that
booking was ultimately cancelled.

## Business Problem(s)
1. **Booking cancellation prediction (primary):** Hotels lose revenue and misjudge staffing
   and overbooking limits when they can't anticipate which reservations will be cancelled.
   A model that flags high-risk bookings lets the hotel intervene early (confirmation
   emails, deposit requests, smarter overbooking).
2. **Demand forecasting (secondary):** Aggregating bookings by month/hotel type can support
   forecasting future demand for revenue management and staffing.

## ML Problem Framing
- **Type:** Binary **Classification**.
- **Justification:** The primary business question — "will this booking be cancelled?" — has
  a binary yes/no answer, so it maps directly to a classification model
  (e.g., Logistic Regression, Random Forest, Gradient Boosting/XGBoost), evaluated with
  precision/recall/ROC-AUC rather than plain accuracy, since cancellations are the minority
  class (~25% of bookings).
- The secondary problem (demand forecasting) would instead be framed as **regression /
  time-series forecasting**, predicting a numeric count of bookings per time period.

## Target Variable & Key Features
- **Target variable:** `is_canceled` (0 = booking kept, 1 = booking cancelled)
- **Key features:**
  - `lead_time` – days between booking and arrival
  - `deposit_type` – No Deposit / Non Refund / Refundable
  - `market_segment` / `distribution_channel` – how the booking was made
  - `customer_type` – Transient, Group, Contract, etc.
  - `previous_cancellations` / `is_repeated_guest`
  - `total_of_special_requests`, `required_car_parking_spaces`
  - `adr` (average daily rate)
  - `booking_changes`

## Three Key Observations
1. **Class imbalance:** Roughly a quarter of bookings are cancelled — not extreme, but
   enough that accuracy alone would be a misleading evaluation metric.
2. **Lead time matters a lot:** Bookings made far in advance are cancelled noticeably more
   often than last-minute bookings, making `lead_time` one of the strongest predictive
   signals.
3. **Missingness is informative, not random:** `company` and `agent` are missing for most
   rows simply because most bookings have no associated company or travel agent — this
   should be encoded (e.g., a `has_company` flag) rather than dropped or blindly imputed.
   `deposit_type` also shows a counter-intuitive pattern (Non Refund bookings cancel *more*
   often than No Deposit ones), which is worth investigating rather than taking at face
   value.

## Repository Contents
- `analysis.ipynb` — dataset exploration using Pandas (shape, dtypes, missing values,
  summary statistics, and the observations above)
- `README.md` — this file

