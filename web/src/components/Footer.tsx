export default function Footer() {
	const commitHash = __COMMIT_HASH__;
	const buildTime = new Date(__BUILD_TIME__).toLocaleString();

	return (
		<footer className="mt-16 text-center text-sm text-white/60">
			<p>
				Build {commitHash} • {buildTime}
			</p>
		</footer>
	);
}
