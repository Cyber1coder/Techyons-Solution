document.addEventListener("DOMContentLoaded", function () {
  const submitBtn = document.getElementById("submitBtn");
  const successMsg = document.getElementById("successMsg");
  const resultsContainer = document.getElementById("resultsContainer");

  // Inputs
  const baseSalaryInput = document.getElementById("baseSalary");
  const avgAnnualStockGrantInput = document.getElementById("avgAnnualStockGrantValue");
  const avgAnnualBonusInput = document.getElementById("avgAnnualBonusValue");
  const totalCompInput = document.getElementById("totalCompensation");

  // Live total compensation calculation
  function calculateTotalCompensation() {
    const base = parseFloat(baseSalaryInput.value) || 0;
    const stock = parseFloat(avgAnnualStockGrantInput.value) || 0;
    const bonus = parseFloat(avgAnnualBonusInput.value) || 0;
    const total = base + stock + bonus;
    totalCompInput.value = total.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 2 });
  }

  baseSalaryInput.addEventListener("input", calculateTotalCompensation);
  avgAnnualStockGrantInput.addEventListener("input", calculateTotalCompensation);
  avgAnnualBonusInput.addEventListener("input", calculateTotalCompensation);

  // Submit form handler
  submitBtn.addEventListener("click", async function (e) {
    e.preventDefault();

    // Elements
    const company = document.getElementById("company").value.trim();
    const title = document.getElementById("title").value.trim();
    const jobFamily = document.getElementById("jobFamily").value;
    const level = document.getElementById("level").value.trim();
    const focusTag = document.getElementById("focusTag").value.trim();
    const education = document.getElementById("education").value;
    const compPerspective = document.querySelector('input[name="compPerspective"]:checked').value;
    const offerMonth = document.getElementById("offerMonth").value;
    const offerYear = document.getElementById("offerYear").value;
    const yearsOfExperience = parseInt(document.getElementById("yearsOfExperience").value) || 0;
    const yearsAtCompany = parseInt(document.getElementById("yearsAtCompany").value) || 0;
    const yearsAtLevel = parseInt(document.getElementById("yearsAtLevel").value) || 0;
    const location = document.getElementById("location").value.trim();
    const workArrangement = document.getElementById("workArrangement").value;
    const employmentType = document.getElementById("employmentType").value;
    const gender = document.getElementById("gender").value;
    const baseSalary = parseFloat(baseSalaryInput.value) || 0;
    const baseSalaryCurrency = document.getElementById("baseSalaryCurrency").value;
    const avgAnnualStockGrantValue = parseFloat(avgAnnualStockGrantInput.value) || 0;
    const avgAnnualBonusValue = parseFloat(avgAnnualBonusInput.value) || 0;
    const annualTargetBonusPercentage = parseFloat(document.getElementById("annualTargetBonusPercentage").value) || 0;

    // Validate required fields
    if (!company || !title || !jobFamily || !level || !location || !baseSalaryInput.value) {
      alert("Please fill all required fields: Company, Title, Job Family, Level, Location, and Base Salary.");
      return;
    }

    // Set loading state
    submitBtn.disabled = true;
    submitBtn.innerText = "Validating...";
    resultsContainer.style.display = "none";

    // Format offerDate to match dataset schema: e.g. "Mon May 15 2025 19:17:13 GMT+0000 (UTC)"
    const dayNames = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const monthNames = {
      Jan: 0, Feb: 1, Mar: 2, Apr: 3, May: 4, Jun: 5,
      Jul: 6, Aug: 7, Sep: 8, Oct: 9, Nov: 10, Dec: 11
    };
    
    // Construct approximate date
    const d = new Date();
    d.setFullYear(parseInt(offerYear));
    d.setMonth(monthNames[offerMonth] || 4);
    d.setDate(15); // middle of month
    
    const dayName = dayNames[d.getDay()];
    const fullMonthName = d.toLocaleString('en-US', { month: 'short' });
    const dateStr = `${dayName} ${fullMonthName} 15 ${offerYear} 09:00:00 GMT+0000 (UTC)`;

    // Calculate total compensation value
    const totalCompensation = baseSalary + avgAnnualStockGrantValue + avgAnnualBonusValue;

    // Package data matching the dataset format
    const formData = {
      company: company,
      title: title,
      jobFamily: jobFamily,
      level: level,
      focusTag: focusTag || jobFamily,
      yearsOfExperience: yearsOfExperience,
      yearsAtCompany: yearsAtCompany,
      yearsAtLevel: yearsAtLevel,
      offerDate: dateStr,
      location: location,
      workArrangement: workArrangement,
      compPerspective: compPerspective,
      baseSalary: baseSalary,
      baseSalaryCurrency: baseSalaryCurrency,
      employmentType: employmentType,
      totalCompensation: totalCompensation,
      avgAnnualStockGrantValue: avgAnnualStockGrantValue,
      avgAnnualBonusValue: avgAnnualBonusValue,
      gender: gender || null,
      ethnicity: null, // Default
      education: education,
      annualTargetBonusPercentage: annualTargetBonusPercentage,
      userCurrency: baseSalaryCurrency
    };

    try {
      const response = await fetch("http://localhost:5000/classify", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        throw new Error("Server error during validation.");
      }

      const result = await response.json();

      // Show success popup momentarily
      successMsg.style.display = "block";
      setTimeout(() => {
        successMsg.style.display = "none";
      }, 3000);

      // Render Verdict Badge
      const verdictBadge = document.getElementById("verdictBadge");
      const decision = result.decision; // "Approve" | "Reject" | "Review"
      
      verdictBadge.innerText = decision === "Approve" ? "Approved / Inlier" : decision === "Reject" ? "Rejected / Outlier" : "Human Review Required";
      
      if (decision === "Approve") {
        verdictBadge.style.background = "#10b981"; // Green
      } else if (decision === "Reject") {
        verdictBadge.style.background = "#ef4444"; // Red
      } else {
        verdictBadge.style.background = "#f59e0b"; // Orange
      }

      // Render Scores
      document.getElementById("resConsistencyScore").innerText = `${(result.consistency_score * 100).toFixed(1)}%`;
      document.getElementById("resTrustScore").innerText = result.user_trust.toFixed(2);

      // Render Flag Reasons
      const reasonsBlock = document.getElementById("reasonsBlock");
      const reasonsList = document.getElementById("reasonsList");
      reasonsList.innerHTML = "";

      if (result.explanation && result.explanation.trim() !== "") {
        reasonsBlock.style.display = "block";
        const flags = result.explanation.split(";");
        flags.forEach(flag => {
          if (flag.trim() !== "") {
            const li = document.createElement("li");
            li.innerText = flag.trim();
            reasonsList.appendChild(li);
          }
        });
      } else {
        reasonsBlock.style.display = "none";
      }

      // Render Segment Statistics
      document.getElementById("resSegmentName").innerText = result.segment || "N/A";
      document.getElementById("resSegmentSize").innerText = result.segment_size || "0";
      
      if (result.segment_avg_salary) {
        document.getElementById("resSegmentAvg").innerText = `${result.segment_avg_salary.toLocaleString(undefined, { maximumFractionDigits: 0 })} USD`;
      } else {
        document.getElementById("resSegmentAvg").innerText = "N/A";
      }

      // Reveal results block
      resultsContainer.style.display = "block";
      resultsContainer.scrollIntoView({ behavior: "smooth" });

    } catch (error) {
      console.error(error);
      alert("Error contacting the backend validation server. Please make sure the backend is running.");
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerText = "Submit & Validate Salary";
    }
  });
});
