# AI-Rewritten Disclosure Q&A Answer

## 60-second oral English answer

This is a real limitation, and we tested it. Our baseline is transparent and explainable, but keyword signals are attackable because AI can rewrite ESG claims using different wording. To address this, we added 5 semantic features and ran a 15-case adversarial rewrite test. The result was clear: keyword signal dropped on average 1.00, while semantic signal increased by 1.60. At the same time, specificity gap increased by 2.93, meaning rewritten disclosures became vaguer. This suggests a keyword-only model would be fragile, but semantic features can still capture meaning-level ESG language and vague commitment patterns. We also rely on Walk / External evidence, so the model never trusts company wording alone. It combines disclosure language with external ESG risk signals and sector-relative context. Finally, the model is a risk screening tool for further due diligence, not a legal accusation. It helps analysts decide which companies need deeper ESG review.

## 中文同義對照版

這確實是一個真實限制，而且我們有測試。原本的 baseline 很透明、容易解釋，但 keyword 訊號可能被攻擊，因為 AI 可以用不同文字改寫 ESG 說法。為了處理這點，我們新增 5 個 semantic features，並做了 15 個案例的 adversarial rewrite test。結果顯示：keyword signal 平均下降 1.00，但 semantic signal 反而增加 1.60；同時 specificity gap 增加 2.93，代表改寫後的揭露變得更空泛。這表示只靠 keyword 的模型會比較脆弱，但 semantic features 仍能抓到語意層級的 ESG 說法與空泛承諾。我們也依賴 Walk / External evidence，所以模型不會只相信公司自己的文字。它會結合揭露語言、外部 ESG 風險訊號與同產業脈絡。最後，這個模型是 ESG 查核優先順序的風險篩選工具，不是法律指控。
